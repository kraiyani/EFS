import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# Read data csv files
def read_data():
    logging.info("Entered into the data ingestion method ... reading CSVs")
    try:
        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the relative file path
        Client_Bureau_Information_file_path = os.path.join(script_directory, '..', '..', 'notebook', 'raw_data', 'Client_Bureau_Information.csv')
        Client_Information_file_path = os.path.join(script_directory, '..', '..', 'notebook', 'raw_data', 'Client_Information.csv')
        Default_Data_file_path = os.path.join(script_directory, '..', '..', 'notebook', 'raw_data', 'Default_Data.csv')
        Loan_Information_file_path = os.path.join(script_directory, '..', '..', 'notebook', 'raw_data', 'Loan_Information.csv')

        # Print the resulting paths

        df_client_burea = pd.read_csv(Client_Bureau_Information_file_path, delimiter=',', low_memory=False)
        df_client_info = pd.read_csv(Client_Information_file_path, delimiter=',', low_memory=False)
        df_default_data = pd.read_csv(Default_Data_file_path, delimiter=',', low_memory=False)
        df_loan_info = pd.read_csv(Loan_Information_file_path, delimiter=',', low_memory=False)

        logging.info("CSVs reading done")

        return df_client_burea, df_client_info, df_default_data, df_loan_info

    except Exception as e:
        raise CustomException(e,sys)
    

