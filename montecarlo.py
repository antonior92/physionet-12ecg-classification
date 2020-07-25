import os
import optuna

main_path = os.getcwd()

def execute(trial):
    i = trial._trial_id
    #pretrain comand setup
    pretrain_setup = ('python pretrain.py --cuda --folder "{}/outputs_gridsearch/iteration{}"'.format(main_path,i),
     '--pretrain_model {}'.format(trial.suggest_categorical('pretrain_model',['LSTM', 'GRU', 'RNN', 'Transformer', 'Transformer XL'])))
    
    pretrain_cmd = " ".join(pretrain_setup)
    print(pretrain_cmd)
    #os.system(pretrain_cmd)






search_space = {'pretrain_model': ['LSTM', 'GRU', 'RNN', 'Transformer', 'Transformer XL']
}

study = optuna.create_study(sampler = optuna.samplers.GridSampler(search_space) , direction = 'maximize' )
study.optimize(execute)
