import os,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Initiating model training")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'AdaBoostRegressor' : AdaBoostRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor()
            }     

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n===============================================================================================\n")
            logging.info(f"Model Report: {model_report}")

            ## to get best model out of the dictionary
            best_model_score=max(sorted(model_report.values())) 
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            ## saving the best model
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            logging.info("Model Training Completed Successfully")

        except Exception as e:
            logging.error(f"Exception occured while training the model : {e}")
            raise CustomException(f"Exception occured while training the model : {e}")