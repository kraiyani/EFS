import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_preprocessing import preprocessor
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# For each file read, preprocessing
def fusion_and_preprocessor():
    logging.info("Entered into the data fusion and processing method ... preparing one CSV")
    try:
        
        # getting data from data_preprocessing module
        df_client_burea, df_client_info, df_default_data, df_loan_info = preprocessor()

        # merging loan with default
        df_loan_default = pd.merge(df_loan_info,df_default_data, on = ['UID', 'RECORDNUMBER'])
    
        # mergning burea and loan_default
        df_burea_loan_default = pd.merge(df_client_burea,df_loan_default, on = ['UID'])

        # merging client and burea_loan_default
        df_client_burea_loan_default = pd.merge(df_client_info,df_burea_loan_default, on = ['UID'])

        # dropping unwanted columns
        df_client_burea_loan_default = df_client_burea_loan_default.drop(columns=['UID', 'REPAYPERIOD'])

        # Drop columns with more than 70% missing values
        threshold = 0.7 * len(df_client_info)
        df_client_burea_loan_default = df_client_burea_loan_default.dropna(axis=1, thresh=threshold)
        
        # filling remaining null with column average
        df_client_burea_loan_default = df_client_burea_loan_default.fillna(df_client_burea_loan_default.mean())

        # temparory storing file
        script_directory = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(script_directory, '..', '..', 'notebook', 'raw_data', 'df_client_burea_loan_default.csv')
        df_client_burea_loan_default.to_csv(temp_path,index=False,sep=',')

        # reading it back with dtypes value
        df_client_burea_loan_default = pd.read_csv(temp_path)

        # drop all the column with mix data types
        df_client_burea_loan_default = df_client_burea_loan_default.drop(columns=list(df_client_burea_loan_default.select_dtypes(include='object')))

        logging.info("fusion and processing into one CSV done")

        return df_client_burea_loan_default

    except Exception as e:
        raise CustomException(e,sys)
    
