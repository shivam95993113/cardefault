from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re

# Assuming the existence of these custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own',
                                'Client_Gender', 'Client_Housing_Type', 'Client_Occupation', 'Type_Organization',
                                'Client_Marital_Status', 'Loan_Contract_Type', 'Homephone_Tag', 'Client_Contact_Work_Tag']
            numerical_cols = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 
                              'Employed_Days', 'Registration_Days', 'ID_Days', 'Population_Region_Relative',
                              'Credit_Bureau', 'Child_Count']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder()),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def clean_special_characters(self, df):
        try:
            logging.info("Cleaning special characters from the dataset")

            # Define columns to clean special characters
            columns_to_clean = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 
                                'Employed_Days', 'Registration_Days', 'ID_Days', 'Population_Region_Relative',
                                'Credit_Bureau', 'Child_Count']

            # Apply regex to remove special characters and handle non-numeric strings
            for column in columns_to_clean:
                df[column] = df[column].astype(str).str.replace(r'[$#@,]', '', regex=True)
                df[column] = pd.to_numeric(df[column], errors='coerce')

            return df
        except Exception as e:
            logging.info("Error occurred during special characters cleaning")
            raise CustomException(e, sys)

    def clean_data(self, df):
        try:
            logging.info("Starting data cleaning process")

            # Clean special characters
            df = self.clean_special_characters(df)

            # Null value analysis before filling missing values
            null_values_before = df.isnull().sum()
            logging.info(f"Null Values Before Filling:\n{null_values_before[null_values_before > 0]}")

            # Define strategies for filling missing values
            columns_to_fill_median = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 'Employed_Days', 
                                      'Registration_Days', 'ID_Days', 'Population_Region_Relative', 'Credit_Bureau', 'Child_Count']
            columns_to_fill_mode = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 
                                    'Client_Gender', 'Client_Housing_Type', 'Client_Occupation', 'Type_Organization',
                                    'Client_Marital_Status', 'Loan_Contract_Type', 'Homephone_Tag', 'Client_Contact_Work_Tag']
            columns_to_fill_zero = ['Score_Source_1', 'Score_Source_2', 'Score_Source_3', 'Social_Circle_Default']

            # Fill missing values
            for column in columns_to_fill_median:
                df[column].fillna(df[column].median(), inplace=True)

            for column in columns_to_fill_mode:
                df[column].fillna(df[column].mode()[0], inplace=True)

            for column in columns_to_fill_zero:
                df[column].fillna(0, inplace=True)

            # Null value analysis after filling missing values
            null_values_after = df.isnull().sum()
            logging.info(f"Null Values After Cleaning:\n{null_values_after[null_values_after > 0]}")

            return df
        except Exception as e:
            logging.info("Error occurred during data cleaning process")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data with low_memory=False to avoid dtype warnings
            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Starting data cleaning process for train and test datasets')

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            logging.info('Data cleaning completed')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Default'  # Replace with your actual target variable
            drop_columns = [target_column_name, 'ID', 'Own_House_Age', 'Score_Source_1', 'Social_Circle_Default', 'Score_Source_3']  # Adjust according to your dataset

            # Features into independent and dependent features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Processor pickle is created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)

