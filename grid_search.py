import os  
import pandas #ler arquivo csv
import optuna 
import csv
import joblib
import shutil
import subprocess as subp

#define paths 

main_path = 'C:\\Users\\danie\\OneDrive\\Área de Trabalho\\IC\\Projetos\\Physionet Challenge\\physionet-12ecg-classification'
dump_file_path = 'C:\\Users\\danie\\OneDrive\\Área de Trabalho\\optuna test\\12ECG_optuna.pkl'
os.chdir(main_path)


def execute(trial):
    i = trial._trial_id
    if os.path.isdir("".join((main_path,'\\outputs gridsearch'))) == False:
        os.mkdir('outputs gridsearch')
    os.chdir("".join((main_path,'\\outputs gridsearch')))
    #create directory for each iteration with a specific name that can later be accessed( removed because train.py already creates it)
    #if os.path.exists('iteretion{}'.format(i)):
    #    shutil.rmtree('iteretion{}'.format(i))
    #os.makedirs('iteretion{}'.format(i))


    os.chdir(main_path)
    #pretrain.py setup
    pre_set_up = ('python pretrain.py --folder "{}\\outputs gridsearch\\iteration{}"'.format(main_path,i),
        '--lr {}'.format(trial.suggest_loguniform('pre_lr', 0.0001, 1)), 
        '--lr_factor {}'.format(trial.suggest_loguniform('pre_lr_factor',0.0001,1)), 
        '--dropout {}'.format(trial.suggest_float('pre_dropout_rate', 0.001, 1.0)),
        '--num_heads {}'.format(trial.suggest_int('pre_num_heads', 5, 11)))
    #train.py setup
    train_set_up =('python train.py --n_total 12 --epochs 10 --folder "{}\\outputs gridsearch\\iteration{}"'.format(main_path,i),
        '--kernel_size {}'.format(trial.suggest_int('kernel_size', 3, 36)), 
        '--dropout_rate {}'.format(trial.suggest_float('dropout_rate', 0.001, 1.0)),
        '--lr_factor {}'.format(trial.suggest_loguniform('lr_factor',0.0001,1)),
        '--lr {}'.format(trial.suggest_loguniform('lr', 0.0001, 1)) ,
        '--batch_size {}'.format(trial.suggest_int('batch_size', 15, 300)))
    
    #pretrain comand
    pre_cmd = " ".join(pre_set_up)
    subp.check_call(pre_cmd, shell=True)
    #train command
    train_cmd = " ".join(train_set_up)
    subp.check_call(train_cmd, shell=True)
    #os.system(cmd) caso falhe
    #reads csv file
    geom_mean = geom_mean_searcher(i)

    return(geom_mean)
    

#searches for the best geom_mean in a given csv file
def geom_mean_searcher(i):
    history_path="{}\\outputs gridsearch\\iteration{}\\history.csv".format(main_path,i)
    best_geom_mean=0
    with open(history_path,'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            if float(line['geom_mean']) > best_geom_mean:
                best_geom_mean = float(line['geom_mean'])
    return(best_geom_mean)

#defining search space
search_space = {
    'pre_lr':[0.01,0.001],
    'pre_lr_factor':[0.1],
    'pre_dropout_rate':[0.1, 0.2 , 0.3],
    'pre_num_heads':[6,8,10],
    'kernel_size': [11, 17, 35],
    'dropout_rate': [0.3, 0.5, 0.7 ],
    'lr_factor': [0.1],
    'lr': [0.001, 0.01],
    'batch_size': [32 ,64]
}
#calculates number of trials
number_trials=1
for key, value in search_space.items() :
    number_trials *= len(value)

#sets up the study and calls the function
study= optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
study.optimize(execute, n_trials = number_trials)
joblib.dump(study, dump_file_path)

#afterexecuting shows best parameters
print('\n',study.best_trial)
print('\nParams with the best results:',study.best_params)
print('\nThe best value was:',study.best_value)


#todo: 
#testar hiperparametros
#adicionar pretreino
#deletar modelos para não estourar memória
#argparse para simplificar codigo


"""
Done:
figure how to wait until the csv file is complete to run the file (line 35) -> aparently os.system
    already waits(have to check when runing)
define the parameters range for the optimization
figure out how to pass i as a working argument (used trial._trial_id to solve problem)
finish writing the code to show the results for each parameter and the best result found
#fix .join's
#testar codigo
"""