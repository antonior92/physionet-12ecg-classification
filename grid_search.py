import os  
import pandas #ler arquivo csv
import optuna 
import csv
import joblib
import shutil
import subprocess as subp

#define paths 

main_path = 'C:\\Users\\danie\\OneDrive\\Área de Trabalho\\IC\\Projetos\\Physionet Challenge\\physionet-12ecg-classification'
#dump_file_path = 'C:\\Users\\danie\\OneDrive\\Área de Trabalho\\optuna test\\12ECG_optuna.pkl'
os.chdir(main_path)


def execute(trial):
    i = trial._trial_id
    if i != 0 :
        del_unworthy_trials(os.path.join(main_path,'outputs gridsearch'), study.best_trial.number)
    
    #create directory for each iteration with a specific name that can later be accessed(removed because pretrain.py and train.py already creates it)
    #os.chdir("".join((main_path,'\\outputs gridsearch')))
    #if os.path.exists('iteretion{}'.format(i)):
    #    shutil.rmtree('iteretion{}'.format(i))
    #os.makedirs('iteretion{}'.format(i))
    #os.chdir(main_path)
    
    #pretrain.py setup
    pre_set_up = ('python pretrain.py --n_total 12 --epochs 10 --folder "{}\\outputs gridsearch\\iteration{}"'.format(main_path,i),
        '--lr {}'.format(trial.suggest_loguniform('pre_lr', 0.0001, 1)), 
        '--lr_factor {}'.format(trial.suggest_loguniform('pre_lr_factor',0.0001,1)), 
        '--dropout {}'.format(trial.suggest_float('pre_dropout_rate', 0.001, 1.0)),
        '--num_heads {}'.format(trial.suggest_int('pre_num_heads', 1, 11)))
    #train.py setup
    train_set_up =('python train.py --n_total 12 --epochs 10 --folder "{}\\outputs gridsearch\\iteration{}"'.format(main_path,i),
        '--kernel_size {}'.format(trial.suggest_int('kernel_size', 3, 36)), 
        '--dropout_rate {}'.format(trial.suggest_float('dropout_rate', 0.001, 1.0)),
        '--lr_factor {}'.format(trial.suggest_loguniform('lr_factor',0.0001,1)),
        '--lr {}'.format(trial.suggest_loguniform('lr', 0.0001, 1)) ,
        '--batch_size {}'.format(trial.suggest_int('batch_size', 15, 300)),
        '--emb_size {}'.format(trial.suggest_int('pre_emb_size',25,100)))
    
    #pretrain comand
    pre_cmd = " ".join(pre_set_up)
    #subp.check_call(pre_cmd, shell=True)s
    os.system(pre_cmd)
    #train command
    train_cmd = " ".join(train_set_up)
    #subp.check_call(train_cmd, shell=True)
    os.system(train_cmd)    
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

#function to delete models that werent best in order to avoid memory overload
def del_unworthy_trials(dir_name, best_iter):
    test = os.listdir(dir_name)

    for item in test:
        if item != "iteration{}".format(best_iter):
         for file in os.listdir(os.path.join(dir_name,item)):  
            if file.endswith(".pth"):
                 os.remove(os.path.join(dir_name, item, file))
    return

#defining search space
search_space = {
    'pre_lr':[0.01],
    'pre_lr_factor':[0.1],
    'pre_dropout_rate':[0.1, 0.2],
    'pre_num_heads':[2,5],
    'pre_emb_size':[50],
    'kernel_size': [17],
    'dropout_rate': [0.3],
    'lr_factor': [0.1],
    'lr': [0.01],
    'batch_size': [32]
}
#creates gridsearch folder
if os.path.isdir("".join((main_path,'\\outputs gridsearch'))) == False:
        os.mkdir('outputs gridsearch')

#calculates number of trials
number_trials=1
for key, value in search_space.items() :
    number_trials *= len(value)

#sets up the study and calls the function
study= optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
study.optimize(execute, n_trials = number_trials)
#joblib.dump(study, dump_file_path)

#delete the model of the last iteration if it isnt the best value
del_unworthy_trials(os.path.join(main_path,'outputs gridsearch'), study.best_trial.number)

#after executing shows best parameters
print('\nBest study trial: ',study.best_trial)
print('\nParams with the best results:',study.best_params)
print('\nThe best value was:',study.best_value)


#todo: 
#deletar modelos para não estourar memória
    #posso usar study.best_trial.number para remover modelo das pastas que não forem o best trial. Adicionar código na linha 53.
#argparse para simplificar codigo
#codigo não funciona quando a pasta grid search tem iterações antigas nela

"""
Done:
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