import os  
import pandas #ler arquivo csv
import optuna 
import csv
import joblib
import shutil
import subprocess as subp

#define paths 

main_path = os.getcwd()
#dump_file_path = 'C:\\Users\\danie\\OneDrive\\Área de Trabalho\\optuna test\\12ECG_optuna.pkl'
os.chdir(main_path)


def execute(trial):
    i = trial._trial_id
    if i != 0 :
        del_unworthy_trials(os.path.join(main_path,'outputs_gridsearch'), study.best_trial.number)
    
    #create directory for each iteration with a specific name that can later be accessed(removed because pretrain.py and train.py already creates it)
    #os.chdir("".join((main_path,'\\outputs_gridsearch')))
    #if os.path.exists('iteretion{}'.format(i)):
    #    shutil.rmtree('iteretion{}'.format(i))
    #os.makedirs('iteretion{}'.format(i))
    #os.chdir(main_path)
    
    #pretrain.py setup
    '''pre_set_up = ('python pretrain.py --cuda --folder "{}/outputs_gridsearch/iteration{}"'.format(main_path,i),
        '--lr {}'.format(trial.suggest_loguniform('pre_lr', 0.0001, 1)), 
        '--lr_factor {}'.format(trial.suggest_loguniform('pre_lr_factor',0.0001,1)), 
        '--dropout {}'.format(trial.suggest_float('pre_dropout_rate', 0.001, 1.0)),
        '--num_heads {}'.format(trial.suggest_int('pre_num_heads', 1, 11)))'''
    #train.py setup
    pred_stage_type = trial.suggest_categorical('pred_stage_type',['lstm', 'gru', 'rnn','mean' ,'max'])
    if pred_stage_type in ('rnn','lstm','gru'):
        train_set_up =('python train.py --cuda --valid_classes dset --train_classes dset --folder "{}/outputs_gridsearch/iteration{}"'.format(main_path,i),
            '--kernel_size {}'.format(trial.suggest_categorical('kernel_size', [9,15,17,35])), 
            '--dropout_rate {}'.format(trial.suggest_float('dropout_rate', 0.001, 1.0)),
            '--out_layer {}'.format(trial.suggest_categorical('out_layer',['sigmoid','softmax'])),
            '--lr {}'.format(trial.suggest_loguniform('lr',0.001,0.01)),
            '--pred_stage_type {}'.format(pred_stage_type),
            '--pred_stage_hidd {}'.format(trial.suggest_categorical('pred_stage_hidd',[30, 400, 800, 1200]))  
            )
    else:
        train_set_up =('python train.py --cuda --valid_classes dset --train_classes dset --folder "{}/outputs_gridsearch/iteration{}"'.format(main_path,i),
        '--kernel_size {}'.format(trial.suggest_int('kernel_size', 3, 36)), 
        '--dropout_rate {}'.format(trial.suggest_float('dropout_rate', 0.001, 1.0)),
        '--out_layer {}'.format(trial.suggest_categorical('out_layer',['sigmoid','softmax'])),
        '--lr {}'.format(trial.suggest_int('lr',0.001,0.01,0.001)),
        '--pred_stage_type {}'.format(pred_stage_type)  
        )

    #pretrain comand
    #pre_cmd = " ".join(pre_set_up)
    #subp.check_call(pre_cmd, shell=True)s
    #os.system(pre_cmd) primeiro teste na gpu do dce
    
    #train command
    train_cmd = " ".join(train_set_up)
    #subp.check_call(train_cmd, shell=True)
    os.system(train_cmd)    
    
    #reads csv file
    geom_mean = geom_mean_searcher(i)

    return(geom_mean)
    

#searches for the best geom_mean in a given csv file
def geom_mean_searcher(i):
    history_path="{}/outputs_gridsearch/iteration{}/history.csv".format(main_path,i)
    best_geom_mean=0
    with open(history_path,'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            if float(line['geom_mean']) > best_geom_mean:
                best_geom_mean = float(line['geom_mean'])
    return(best_geom_mean)

#function to delete models that werent best in order to avoid memory overload
def del_unworthy_trials(dir_name, best_iter):
    test = os.listdir(dir_name)

    for item in test:
        if item != "iteration{}".format(best_iter):
         for file in os.listdir(os.path.join(dir_name,item)):  
            if file.endswith(".pth"):
                 os.remove(os.path.join(dir_name, item, file))
    return


if __name__ == "__main__":
    
#defining search space
#search_space = {
    # 'kernel_size': [9,15,17],
    # 'dropout_rate': [0.2,0.5,0.8]
    #'pre_lr':[0.01],
    #'pre_lr_factor':[0.1],
    #'pre_dropout_rate':[0.2],
    #'pre_num_heads':[2],
    #'pre_emb_size':[50]}

    #creates gridsearch folder
    if os.path.isdir("".join((main_path,'/outputs_gridsearch'))) != False:
            os.mkdir('outputs_gridsearch')

    #calculates number of trials
    # number_trials=1
    # for key, value in search_space.items() :
    #     number_trials *= len(value)

    #sets up the study and calls the function
    study= optuna.create_study(sampler=optuna.samplers.RandomSampler(), direction='maximize') #for gridsearch: .GridSample(search_space)
    study.optimize(execute, n_trials = 25)
    #joblib.dump(study, dump_file_path)

    #delete the model of the last iteration if it isnt the best value
    del_unworthy_trials(os.path.join(main_path,'outputs_gridsearch'), study.best_trial.number)

    #after executing shows best parameters
    print('\nBest study trial: ',study.best_trial)
    print('\nParams with the best results:',study.best_params)
    print('\nThe best value was:',study.best_value)
pass

#todo: 

#argparse para simplificar codigo
#codigo não funciona quando a pasta grid search tem iterações antigas nela

"""
Done:
#deletar modelos para não estourar memória
    #posso usar study.best_trial.number para remover modelo das pastas que não forem o best trial. Adicionar código na linha 53.
figure how to wait until the csv file is complete to run the file (line 35) -> aparently os.system
    already waits(have to check when runing)
define the parameters range for the optimization
figure out how to pass i as a working argument (used trial._trial_id to solve problem)
finish writing the code to show the results for each parameter and the best result found
#fix .join's
#testar codigo
#adicionar pretreino
#testar hiperparametros
"""