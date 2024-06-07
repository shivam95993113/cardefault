import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.utils import save_object, load_object, evaluate_model
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    best_model_file_path: str = os.path.join('artifacts', 'best_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, is_classification=True):
        try:
            logging.info("Splitting train and test data into features and target")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            if is_classification:
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest Classifier": RandomForestClassifier()
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor()
                }

            logging.info("Evaluating models")
            model_performance = evaluate_model(X_train, y_train, X_test, y_test, models, is_classification=is_classification)
            logging.info(f"Model performance: {model_performance}")

            if is_classification:
                best_model_name = max(model_performance, key=lambda k: model_performance[k]['F1 Score'])
                best_model_score = model_performance[best_model_name]['F1 Score']
            else:
                best_model_name = max(model_performance, key=lambda k: model_performance[k]['R2 Score'])
                best_model_score = model_performance[best_model_name]['R2 Score']

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            best_model = models[best_model_name]

            logging.info("Saving the best model")
            save_object(self.model_trainer_config.best_model_file_path, best_model)

            return best_model_score

        except Exception as e:
            logging.info("Exception occurred in model trainer")
            raise CustomException(e, sys)

