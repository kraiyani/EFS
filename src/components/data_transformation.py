import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_fusion_and_processing import fusion_and_preprocessor
from src.utils import save_object
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: 
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# For each file read, preprocessing
def transformation():
    logging.info("Entered into the data transformation ... preparing features")
    try:
        
        # getting data from data_fusion_and_processing module
        df_client_burea_loan_default = fusion_and_preprocessor()

        # getting X and Y
        df_x = df_client_burea_loan_default.iloc[:,:-1]
        df_y = df_client_burea_loan_default.iloc[:,-1]

        # normalize
        # define min max scaler
        scaler = MinMaxScaler()

        # Fit the scaler to the DataFrame
        scaler.fit(df_x[df_x.columns])

        # Transform the data
        transformed_data = scaler.transform(df_x[df_x.columns])

        # droping columns with Variance Thresholding 
        var_thres=VarianceThreshold(threshold=0.01)
        constant_columns = [column for column in df_x.columns if column not in df_x.columns[(var_thres.fit(df_x)).get_support()]]
        df_x = df_x.drop(constant_columns,axis=1)

        # finding all features with 80% corelation and dropping them
        corr_features = correlation(df_x, 0.8)
        len(set(corr_features))
        df_x = df_x.drop(corr_features,axis=1)

        # determine the mutual information
        mutual_info = mutual_info_classif(df_x, df_y)

        # Set a threshold for mutual_info_classif values (adjust as needed)
        threshold = 0.1 

        # Identify columns with mutual_info_classif values below the threshold
        low_mi_columns = df_x.columns[mutual_info < threshold]

        # Drop columns with low mutual_info_classif values
        df_x = df_x.drop(columns=low_mi_columns)

        # saving df_x and df_y
        script_directory = os.path.dirname(os.path.abspath(__file__))
        df_x_path = os.path.join(script_directory, '..', '..', 'data', 'df_x.csv')
        df_y_path = os.path.join(script_directory, '..', '..', 'data', 'df_y.csv')
        df_x.to_csv(df_x_path,index=False,sep=',')
        df_y.to_csv(df_y_path,index=False,sep=',')

        # Save the scaler
        temp_df = df_client_burea_loan_default[df_x.columns]
        scaler.fit(temp_df[temp_df.columns])
        pkl_path = os.path.join(script_directory, '..', '..', 'artifacts', 'min_max_scaler.pkl')
        save_object(pkl_path,scaler)

        logging.info("feature preparation done")

        return df_x,df_y

    except Exception as e:
        raise CustomException(e,sys)