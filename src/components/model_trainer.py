import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import transformation
from src.utils import save_object


# Training the best model
def model_train():
    logging.info("Entered into the model trainer ... preparing best model")
    try:

        # getting data from transformation module
        df_x, df_y = transformation()


        X = df_x
        y = df_y

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define a dictionary to store classifier names and corresponding classifiers
         # 'Logistic Regression': LogisticRegression(solver='lbfgs', max_iter=1000),
                    #  'Support Vector Machine': SVC(),
        classifiers = {
           
            'Random_Forest': RandomForestClassifier(),
            'Gradient_Boosting': GradientBoostingClassifier(),
            'K-Nearest_Neighbors': KNeighborsClassifier(),
            'Naive_Bayes': GaussianNB(),
            'Decision_Tree': DecisionTreeClassifier(),
            'Neural_Network': MLPClassifier()
        }

        # Define hyperparameter grids for each classifier
        # 'Logistic Regression': {'C': [0.01, 0.1, 1.0, 10.0]},
        # 'Support Vector Machine': {'C': [0.01, 0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
        param_grids = {
            'Random_Forest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]},
            'Gradient_Boosting': {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]},
            'K_Nearest_Neighbors': {'n_neighbors': [1, 3, 5, 7]},
            'Naive_Bayes': {},
            'Decision_Tree': {'max_depth': [None, 10, 20, 30]},
            'Neural_Network': {'hidden_layer_sizes': [(50,), (100, 50), (100, 100)]}
        }

        # Create a dictionary to store the best classifiers and their corresponding parameters
        best_classifiers = {}

        # Iterate through classifiers and perform GridSearchCV
        for clf_name, clf in classifiers.items():
            param_grid = param_grids.get(clf_name, {})  # Get the corresponding parameter grid
            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Get the best classifier and its parameters
            best_classifier = grid_search.best_estimator_
            best_parameters = grid_search.best_params_
            
            # Store the best classifier and parameters in the dictionary
            best_classifiers[clf_name] = {'best_classifier': best_classifier, 'best_parameters': best_parameters}

        # Evaluate the best classifiers on the test set
        results = []

        for clf_name, clf_info in best_classifiers.items():
            best_classifier = clf_info['best_classifier']

            # Save the best models 
            script_directory = os.path.dirname(os.path.abspath(__file__))
            pkl_path = os.path.join(script_directory, '..', '..', 'artifacts', clf_name+'.pkl')
            save_object(pkl_path,best_classifier)

            y_pred = best_classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Calculate R-squared (r2_score)
            r2 = r2_score(y_test, y_pred)  

            results.append([clf_name, accuracy, precision, recall, f1, r2])  

        # creating results df and saving it
        results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'R-squared'])
        file_path = os.path.join(script_directory, '..', '..', 'data', 'results_df.csv')
        results_df.to_csv(file_path,index=False)

        logging.info("modeling traning done")
        
        return results_df

    except Exception as e:
        raise CustomException(e,sys)


