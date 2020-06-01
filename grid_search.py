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
    #create directory for each iteration with a specific name that can later be accessed, already created
    #on train.py
    #if os.path.exists('iteretion{}'.format(i)):
    #    shutil.rmtree('iteretion{}'.format(i))
    #os.makedirs('iteretion{}'.format(i))


    os.chdir(main_path)
    setup =('python train.py --epochs 10 --folder "{}\\outputs gridsearch\\iteration{}"'.format(main_path,i),
        '--kernel_size {}'.format(trial.suggest_int('kernel_size', 3, 35)), 
        '--dropout_rate {}'.format(trial.suggest_float('dropout_rate', 0.001, 1.0)),
        '--lr_factor {}'.format(trial.suggest_loguniform('lr_factor',0.0001,1)),
        '--lr {}'.format(trial.suggest_loguniform('lr', 0.0001, 1)) ,
        '--batch_size {}'.format(trial.suggest_int('batch_size', 15, 300)))

    cmd = " ".join(setup)
    os.system(cmd)
    #subp.check_call(cmd, shell=True)
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
    'kernel_size': [5, 10, 17, 34],
    'dropout_rate': [0.3, 0.5, 0.7 ],
    'lr_factor': [0.1],
    'lr': [0.001, 0.01],
    'batch_size': [32 ,64]
}


#sets up the study and calls the function
study= optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
study.optimize(execute, n_trials=4*3*1*2*2)
joblib.dump(study, dump_file_path)

#afterexecuting shows best parameters
print(study.best_trial)
print('Params with the best results',study.best_params)
print('The best value was:',study.best_value)


#todo: 
#fix .join's
#argparse para simplificar codigo
#testar codigo

"""
Done:
figure how to wait until the csv file is complete to run the file (line 35) -> aparently os.system
    already waits(have to check when runing)
define the parameters range for the optimization
figure out how to pass i as a working argument (used trial._trial_id to solve problem)
finish writing the code to show the results for each parameter and the best result found
"""