import os 
import sys
import dill
import pickle

import numpy as np
import pandas as pd

from exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    
    except Exception as e:
        raise CustomException(e,sys)
    
    

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name= (list(models.keys())[i])
            model = (list(models.values())[i])
            param= params[list(models.keys())[i]]
            gs= GridSearchCV(model,param_grid=param,cv=3,n_jobs=-1)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(x_train,y_train)

            score= accuracy_score(y_test,model.predict(x_test))

            report[list(models.keys())[i]]=score
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
        
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)