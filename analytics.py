from data import read_header, CLASSES, multiclass_to_binaryclass
from seaborn.algorithms import bootstrap
import sklearn.metrics as metrics
from seaborn.utils import ci
import pandas as pd
import numpy as np
import argparse
import os.path


def get_analytics_dataframe(output_folder, dataset_folder, train_ids_path, val_ids_path, max_id=6877):
    """This creates a ~big~ ~fat~ DataFrame with the predictions, probabilities and the real outputs.

    DataFrame returned includes:

            'id':   (str) Id corresponding to the file name in the original dataset.
        'pred_*':  (bool) Prediction for each one of the 8 classes.
        'prob_*': (float) Output of the NNet.
       'truth_*':  (bool) Ground truth.
           'age':   (int) Age of the patient.
       'is_male':  (bool) Gender.
      'baseline': (float) Mean baseline.
     'gain_lead': (float) Mean baseline.
          'freq':   (int) Frequency.
    'signal_len':   (int) Signal length.
         'train':  (bool) If row was in the training set.
         'valid':  (bool) If row was in the validation set.

    Example usage:
        df = get_metrics_dataframe("./outputs/", "./Training_WFDB/", "./train_ids.txt", "./valid_ids.txt")
    """

    df_list = []
    for val in range(1, max_id + 1):
        df_tmp = pd.read_csv(os.path.join(output_folder, "A{:04d}.csv".format(val)), comment="#")
        predicted = {"pred_" + key: item.astype(bool) for key, item in dict(df_tmp.loc[0]).items()}
        probs = {"prob_" + key: item for key, item in dict(df_tmp.loc[1]).items()}
        with open(os.path.join(dataset_folder, "A{:04d}.hea".format(val)), "r") as f:
            real = read_header(f.readlines())
            real["output"] = multiclass_to_binaryclass(real["output"])
            for idx, c in enumerate(CLASSES):
                real["truth_" + CLASSES[idx]] = real["output"][idx]
            real.pop("output")
            real["baseline"] = real["baseline"].mean()
            real["gain_lead"] = real["gain_lead"].mean()
        dict_all = {"id": "A{:04d}".format(val)}
        dict_all.update(predicted)
        dict_all.update(probs)
        dict_all.update(real)
        df_list.append(dict_all)

    with open(train_ids_path, "r") as f:
        train_ids = set(f.read().split(","))

    with open(val_ids_path, "r") as f:
        valid_ids = set(f.read().split(","))

    df = pd.DataFrame(df_list)

    df["train"] = df["id"].apply(lambda x: x in train_ids)
    df["valid"] = df["id"].apply(lambda x: x in valid_ids)

    return df


cm_str = \
    """
    pred\\true| False | True |
       False | {:-4}  | {:-4} |
        True | {:-4}  | {:-4} |"""

subset_str = \
    """----------
    -> {} |
    ----------
    
    true support:    {} 
    pred support:    {}
    recall score:    {:3}
    precision score: {:3}
    accuract score:  {:3}
    
    ====================================
    | Confusion Matrix                 |
    ====================================
    |pred\\true| False      | True      |
    |   False | (tn) {:-4}  | (fn) {:-4} |
    |    True | (fp) {:-4}  | (tp) {:-4} |
    ====================================
    """

if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--report_path', type=str, default="./report.txt",
                               help="Path to write the report at.")
    config_parser.add_argument('--output_folder', type=str, default="./outputs/",
                               help="Path with the model outputs.")
    config_parser.add_argument('--dataset_folder', type=str, default="./Training_WFDB/",
                               help="Path with the training data.")
    config_parser.add_argument('--train_ids_path', type=str, default="./train_ids.txt",
                               help="Path with the train ids used by the model.")
    config_parser.add_argument('--val_ids_path', type=str, default="./valid_ids.txt",
                               help="Path with the test ids used by the model.")

    args = config_parser.parse_args()

    df = get_analytics_dataframe(args.output_folder, args.dataset_folder, args.train_ids_path, args.val_ids_path)

    with open(args.report_path, "w") as f:
        pass

    for c in CLASSES:
        print(c)
        pred = df["pred_{}".format(c)].values
        truth = df["truth_{}".format(c)].values

        strs = []
        for subset in ["train", "valid"]:

            pred_ = pred[df[subset].values]
            truth_ = truth[df[subset].values]
            tp = pred[df[subset].values] & truth[df[subset].values]
            tn = (~pred[df[subset].values]) & (~truth[df[subset].values])
            fp = pred[df[subset].values] & (~truth[df[subset].values])
            fn = (~pred[df[subset].values]) & (truth[df[subset].values])

            strs_cells_all = ""

            cells = [fp + fn, tn + tp, tn, fn, fp, tp]
            cells_names = ["incorrect fp+fn", "correct tn+tp", "tn", "fn", "fp", "tp"]

            for cell, name in zip(cells, cells_names):
                print(name)
                strs_cells = []
                for val in ["age", "signal_len", "is_male"]:
                    stat_array = df[df[subset]][cell][val].values
                    cell_mean = np.mean(stat_array)
                    cell_ci_mean = ci(bootstrap(stat_array, func=np.mean))
                    cell_ci_mean = (cell_ci_mean[1] - cell_ci_mean[0]) / 2
                    cell_median = np.median(stat_array)
                    cell_ci_median = ci(bootstrap(stat_array, func=np.median))
                    cell_ci_median = (cell_ci_median[1] - cell_ci_median[0]) / 2
                    strs_cells.append("mean({}):".format(val).ljust(21) +
                                      " {}+{}".format(round(cell_mean, 2), round(cell_ci_mean, 2)))
                    strs_cells.append("median({}):".format(val).ljust(21) +
                                      " {}+{}".format(round(cell_median, 2), round(cell_ci_median, 2)))
                strs_cells_all += "\n({})\n".format(name) + "\n".join(strs_cells[:-1]) + "\n"

            cm = metrics.confusion_matrix(truth_, pred_)

            strs.append(subset_str.format(subset,
                                          truth_.sum(),
                                          pred_.sum(),
                                          round(metrics.recall_score(truth_, pred_), 3),
                                          round(metrics.precision_score(truth_, pred_), 3),
                                          round(metrics.accuracy_score(truth_, pred_), 3),
                                          cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
                        + strs_cells_all)

        str_final = "=" * 80 + "\n" + c.center(80) + "\n" + "=" * 80 + "\n"

        for linel, liner in zip(strs[0].split("\n"), strs[1].split("\n")):
            str_final += " {} | {} ".format(linel.ljust(37), liner.ljust(37)) + "\n"

        with open(args.report_path, "a") as f:
            f.write(str_final)
