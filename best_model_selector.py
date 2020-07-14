import os
import csv


def GetBestModel(rootdir, model_default_name):
    list_model_paths = []
    best_geom_mean = 0
    best_model_path = 'no models found.'
    #walks through each folder and its files
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #checks for history file
            if (file == 'history.csv'):
                #appends to the list of paths the path to the model the csv file represents
                list_model_paths.append(os.path.join(subdir,model_default_name))
                with open(os.path.join(subdir, file),'r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    next(csv_reader)
                    #iterates over the csv file to find the best geom mean
                    for line in csv_reader:
                        if float(line['geom_mean']) > best_geom_mean:
                            best_geom_mean = float(line['geom_mean'])
                            # if geom mean from csv file is the best register the model related to that history file
                            best_model_path = os.path.join(subdir, model_default_name)

    return(list_model_paths, best_geom_mean, best_model_path)


def DelUnworthyTrials(list_model_paths, best_model):  
    #iterates over list of model paths and deletes the ones that are not the best
    for item in list_model_paths:
        if item != best_model:
            print('\nremoved item: ' + item)
            os.remove(item)
    return




if  __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))
    eval_dir = os.getcwd() + '/evaluation' #enable this lines once the file is in the right folder
  
    model_default_name = 'model.pth' #just a test, choose the right name once we decide it
    #calls function to check best geom mean and find model paths
    model_paths, best_geom_mean, best_model = GetBestModel(eval_dir,model_default_name) 
    
    print('model paths: ',model_paths)
    print('\nbest geometric mean found: ', best_geom_mean)
    print('\nbest model path: ', best_model)
    #delete models that weren't the best
    DelUnworthyTrials(model_paths, best_model)

    pass

"""Todo:
- choose the right name for the model_default_name
- add a completion bar to the process of deleting models
"""

"""Done:
- mudar list_history_paths para salvar modelos
- transform frst part of the code into a  function
- finish del_unworthy_trials function
"""