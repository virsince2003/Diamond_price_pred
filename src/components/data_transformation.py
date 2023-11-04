import os,sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


## Data Transformation config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")



## Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            logging.info("Data Transformation started")
# Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
# Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Data Transformation: Creating pipeline")


            # Create the preprocessing pipelines for both numeric and categorical data
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
            logging.info("Data Transformation: Pipeline created")

        except Exception as e:
            logging.error("Data Transformation: Error while creating pipeline")
            raise CustomException(e,sys)
        



    def initiate_data_transformation(self,train_path,test_path):
            try:
                # Read the train data and test data
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                logging.info("Data Transformation: Data read successfully")
                logging.info(f"Train DataFrame Head: \n {train_df.head().to_string()}")
                logging.info(f"Test DataFrame Head: \n {test_df.head().to_string()}")

                logging.info('Obtaining the preprocessing object')

                preprocessing_obj=self.get_data_transformation_object()
                target_column='price'
                drop_columns=[target_column,'id']

                #features into dependent and independent features
                input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
                target_feature_train_df=train_df[target_column]

                input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
                target_feature_test_df=test_df[target_column]

                ## apply the transformation
                input_feature_train_df=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_df=preprocessing_obj.transform(input_feature_test_df)

                logging.info("Data Transformation: Splitting the data into input and target features")

                train_arr=np.c_[input_feature_train_df,target_feature_train_df]
                test_arr=np.c_[input_feature_test_df,target_feature_test_df]


                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                logging.info("Data Transformation: Preprocessing pickle saved successfully")

                return(
                    train_arr,
                    test_arr
                )
                
            except Exception as e:
                logging.error(f"Data Transformation: Error while initiating data transformation : {e}")
                raise CustomException(e,sys)
            