import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import read_data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# For each file read, preprocessing
def preprocessor():
    logging.info("Entered into the data preprocessing method ... processing individual CSVs")
    try:
        
        # getting data from data_ingestion module
        df_client_burea, df_client_info, df_default_data, df_loan_info = read_data()

        #-----------------
        # default data csv
        #-----------------

        df_default_data['classification_label'] = np.where(df_default_data.DVAL > 0, 1, df_default_data.DVAL) # labelling defaulter as 1
        df_default_data.classification_label.fillna(value=0, inplace=True) # labelling non defaulter as 0
        df_default_data = df_default_data.drop(columns=['DVAL','DMON']) # dropping dval and dmon as they are not needed anymore

        #-----------------
        # loan info csv
        #-----------------

        # Dropping all row with missing values for REPAYPERIOD
        df_loan_info = df_loan_info.dropna(subset=['REPAYPERIOD'])

        # Terms below 12 and over 60 months are out of scope and not eligible
        df_loan_info = df_loan_info[(df_loan_info['REPAYPERIOD'] >= 12) & (df_loan_info['REPAYPERIOD'] <=60 )]

        # checking null values and replaceing them accordingly
        ACCSTARTDATE = df_loan_info[df_loan_info['ACCSTARTDATE'].isnull()].index.tolist()
        FIRST_MONTH = df_loan_info[df_loan_info['FIRST_MONTH'].isnull()].index.tolist()
        LAST_MONTH = df_loan_info[df_loan_info['LAST_MONTH'].isnull()].index.tolist()
        for index in ACCSTARTDATE:
            df_loan_info['ACCSTARTDATE'][index] = df_loan_info['FIRST_MONTH'][index]
        for index in FIRST_MONTH:
            df_loan_info['FIRST_MONTH'][index] = df_loan_info['ACCSTARTDATE'][index]
        for index in LAST_MONTH:
            df_loan_info['LAST_MONTH'][index] = df_loan_info['SEARCHDATE'][index]


        # Creating new features
        df_loan_info['FIRST_MONTH'] = pd.to_datetime(df_loan_info['FIRST_MONTH'])
        df_loan_info['ACCSTARTDATE'] = pd.to_datetime(df_loan_info['ACCSTARTDATE'])
        df_loan_info['LAST_MONTH'] = pd.to_datetime(df_loan_info['LAST_MONTH'])
        df_loan_info['days_without_service'] = (df_loan_info['FIRST_MONTH'] - df_loan_info['ACCSTARTDATE']) / np.timedelta64(1, 'D')
        df_loan_info['days_without_service'] = np.where(df_loan_info['days_without_service'] < 0, 0, df_loan_info['days_without_service'])
        df_loan_info['days_with_service'] = (df_loan_info['LAST_MONTH'] - df_loan_info['FIRST_MONTH']) / np.timedelta64(1, 'D')
        
        # Droping not required columns
        df_loan_info = df_loan_info.drop(columns=['ACCSTARTDATE', 'FIRST_MONTH', 'LAST_MONTH','SEARCHDATE'])

        #-----------------
        # client burea csv
        #-----------------

        df_client_burea['CLASS'] = df_client_burea['CLASS'].map({'STANDARD':0, 'PREMIUM':1})

        #-----------------
        # client info csv
        #-----------------

        # replacing missing values with nan
        df_client_info = df_client_info.replace(-999997.0,np.nan)

        # Dropping duplicate columns
        df_client_info = df_client_info.T.drop_duplicates().T

        # Dropping dataframe columns with only one distinct value
        df_client_info = df_client_info.loc[:,df_client_info.apply(pd.Series.nunique) != 1]

        logging.info("processing individual CSV done")

        return df_client_burea, df_client_info, df_default_data, df_loan_info

    except Exception as e:
        raise CustomException(e,sys)
    