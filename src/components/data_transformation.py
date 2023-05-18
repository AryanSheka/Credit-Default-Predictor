import os 
import sys
from exception import CustomException
from logger import logging

from utils import save_object

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        logging.info("Data Transformation has begun")

        try:
            num_transformer= StandardScaler()
            num_cols=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            num_pipeline= Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",num_transformer)
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_cols)
            ])
            logging.info("Preprocessor is made successfully")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            logging.info("Data Transformation has been initiated")

            train_data= pd.read_csv(train_path)
            test_data= pd.read_csv(test_path)

            logging.info("Getting Preprocessor object")
            preprocessor = self.get_data_transformer_object()

            train_x_input=train_data.drop(columns=['default.payment.next.month','ID'],axis=1)
            test_x_input= test_data.drop(columns=['default.payment.next.month','ID'],axis=1)

            train_y_input=train_data['default.payment.next.month']
            test_y_input= test_data['default.payment.next.month']

            train_x_input_arr= preprocessor.fit_transform(train_data)
            test_x_input_arr= preprocessor.fit_transform(test_data)
            
            logging.info("Train and test data transformed")

            train_arr = np.c_[train_x_input,np.array(train_y_input)]
            test_arr = np.c_[test_x_input_arr,np.array(test_y_input)]

            logging.info("Train and test array created")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Pickle file Created and saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)