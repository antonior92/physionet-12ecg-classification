import warnings
import numpy as np
from ecg_dataset import add_normal_column
from sklearn.metrics import precision_recall_curve
from evaluate_12ECG_score import compute_beta_score


def get_threshold(y_true, y_score, beta=2):
    """Find precision and recall values that maximize f-beta score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_score = (1 + beta**2)* precision * recall / (beta**2*precision + recall)
        # Select threshold that maximize f1 score
        index = np.argmax(f_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


def get_metrics(y_true, y_pred):
    """Return dictionary with relevant metrics"""

    y_true = add_normal_column(y_true)
    y_pred = add_normal_column(y_pred)

    accuracy, f_measure, f_beta, g_beta = compute_beta_score(y_true, y_pred,  num_classes=y_pred.shape[1], beta=2, check_errors=True)

    geometric_mean = np.sqrt(f_beta*g_beta)

    return {'acc': accuracy, 'f_measure': f_measure, 'f_beta': f_beta, 'g_beta': g_beta, 'geom_mean': geometric_mean}
