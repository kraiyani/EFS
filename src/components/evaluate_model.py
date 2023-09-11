import sys
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import model_train
import pandas as pd
import os

# Evaluate the best model
def display_results():
    logging.info("Entered into the evaluation model... finding best score")
    try:

        # getting data from model_trainer module
        # results_df  = model_train()
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, '..', '..', 'data', 'results_df.csv')
        results_df = pd.read_csv(file_path)

        # Initialize an empty metrics_data dictionary
        metrics_data = {}

        # Iterate through the rows of the results DataFrame
        for index, row in results_df.iterrows():
            model_name = row['Classifier']
            metrics_data[model_name] = {
                'Accuracy': row['Accuracy'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1 Score': row['F1 Score'],
                'R-squared': row['R-squared']
            }
            
        # # Return evaluation metrics as a dictionary
        # evaluation_metrics = {
        #     'Accuracy': results_df['Accuracy'].to_list(),
        #     'Precision': results_df['Precision'].to_list(),
        #     'Recall': results_df['Recall'].to_list(),
        #     'F1 Score': results_df['F1 Score'].to_list(),
        #     'R-squared': results_df['R-squared'].to_list()
        # }
        
        logging.info("evaluation done")

        return metrics_data

    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    res = display_results()