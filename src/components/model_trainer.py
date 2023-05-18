import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from dataclasses import dataclass
from exception import CustomException
from logger import logging
from utils import save_object,evaluate_models


@dataclass
class Model_training_config:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_training_config=Model_training_config()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            x_train,y_train,x_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])

            logging.info("The train and test data have been split into x and y")
            
            models= {
                "Ada Boost Classifier": AdaBoostClassifier(),
                "K Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(verbose=0),
                "XGB Classifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier" : SVC()
            }

            params= {
                "Ada Boost Classifier":{
                    "n_estimators":[8,16,32,64,100,128,256],
                    "learning_rate":[0.01,0.01,0.1,0.5,0.7,1],
                },
                "K Neighbors Classifier":{
                    "n_neighbors":[3,5,7],
                    "weights":['distance','uniform'],
                },
                "Decision Tree Classifier":{
                    "criterion":["gini", "entropy", "log_loss"],
                    "splitter":['best','random']
                },
                "Random Forest Classifier":{
                    "n_estimators" :[8,16,32,64,100,128,256],
                    "criterion":["gini", "entropy", "log_loss"],
                    "max_features":['sqrt','log2']
                },
                "XGB Classifier":{
                    "booster":['gbtree','gblinear','dart'],
                    "learning_rate":[0.01,0.05,0.1],
                    'n_estimators': [8,16,32,64,128,256]

                },
                "CatBoosting Classifier":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations': [30, 50, 100]
                },
                'Logistic Regression':{   
                },
                "Support Vector Classifier":{
                }
            }

            logging.info("Evaluating best model and parameters")

            report:dict= evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            logging.info("Models have been tested and accuracy has been recorded")

            best_score= max(sorted(report.values()))

            best_model_name=list(report.keys())[list(report.values()).index(best_score)]
            best_model=models[best_model_name]
            logging.info("Best Model is found to be {}".format(best_model_name))
            logging.info("The Accuracy of the model is {}".format(best_score))
            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved as pickle file")
            return self.model_training_config.trained_model_file_path


        except Exception as e:
            raise CustomException(e,sys)
        