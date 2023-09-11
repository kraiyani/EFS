import os
import sys
import joblib
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:

        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)
