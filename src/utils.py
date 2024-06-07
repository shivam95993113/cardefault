import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
)
import matplotlib.pyplot as plt

def save_object(file_path, obj):
    """
    Save an object to a file.

    Parameters:
    file_path (str): The path where the object should be saved.
    obj (any): The object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file.

    Parameters:
    file_path (str): The path to the file containing the object.

    Returns:
    any: The loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, is_classification=True):
    """
    Train and evaluate models on the provided dataset.

    Parameters:
    X_train (np.ndarray): Training data features.
    y_train (np.ndarray): Training data target.
    X_test (np.ndarray): Testing data features.
    y_test (np.ndarray): Testing data target.
    models (dict): A dictionary where keys are model names and values are the model instances.
    is_classification (bool): Flag to indicate if the problem is classification (True) or regression (False).

    Returns:
    dict: A report of model names and their corresponding test scores.
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            if is_classification:
                # Calculate evaluation metrics for classification models
                accuracy = accuracy_score(y_test, y_test_pred)
                f1 = f1_score(y_test, y_test_pred, average='weighted')
                precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_test_pred, average='weighted')
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                
                # ROC AUC Curve
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Receiver Operating Characteristic for {model_name}')
                plt.legend(loc="lower right")
                plt.savefig(f'artifacts/roc_curve_{model_name}.png')
                plt.close()

                report[model_name] = {
                    'Accuracy': accuracy,
                    'F1 Score': f1,
                    'Precision': precision,
                    'Recall': recall,
                    'ROC AUC': roc_auc
                }
            else:
                # Get R2 score for regression models
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = {'R2 Score': test_model_score}

        return report
    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)