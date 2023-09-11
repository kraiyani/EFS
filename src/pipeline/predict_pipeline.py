import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            
            model_pkl_path = os.path.join(script_directory, '..', '..', 'artifacts', 'Random_Forest.pkl')
            min_max_path = os.path.join(script_directory, '..', '..', 'artifacts', 'min_max_scaler.pkl')

            model=load_object(file_path=model_pkl_path)
            preprocessor=load_object(file_path=min_max_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            proba = model.predict_proba(data_scaled)
            proba = np.round((proba[0][0])*100,2)
            
            if preds[0] == 0:
                result = 'Not a defaul client'
            else:
                result = 'A defaul client'

            return result,proba
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        F_1: int,
        F_3: int,
        F_4: int,
        F_5: int,
        F_6: int,
        F_53: int,
        F_65: int,
        F_260: int,
        days_with_service: int):

        self.F_1 = F_1

        self.F_3 = F_3

        self.F_4 = F_4

        self.F_5 = F_5

        self.F_6 = F_6

        self.F_53 = F_53

        self.F_65 = F_65

        self.F_260 = F_260

        self.days_with_service = days_with_service

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "F_1": [self.F_1],
                "F_3": [self.F_3],
                "F_4": [self.F_4],
                "F_5": [self.F_5],
                "F_6": [self.F_6],
                "F_53": [self.F_53],
                "F_65": [self.F_65],
                "F_260": [self.F_260],
                "days_with_service": [self.days_with_service],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

