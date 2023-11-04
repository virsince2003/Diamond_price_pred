import os,sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            logging.info("Prediction Pipeline started")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)
            return prediction
        
        except Exception as e:
            logging.error(f"Prediction Pipeline failed: {e}")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 cut:str,
                 color:str,
                 clarity:str,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float):
        self.cut=cut
        self.carat=carat
        self.color=color
        self.clarity=clarity
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z]
            }

            df=pd.DataFrame(custom_data_input_dict)
            logging.info("Custom Dataframe created")
            return df
        
        except Exception as e:
            logging.error(f"Custom Dataframe creation failed: {e}")
            raise CustomException(e,sys)
        




