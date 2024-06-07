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

    def replace_unwanted_chars(self, value):
        if isinstance(value, str):
            return re.sub(r'[$#@,]', '', value)
        return value

    def clean_columns(self, data):
        columns_to_clean = [
            'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days',
            'Client_Income', 'Credit_Amount', 'Loan_Annuity',
            'Population_Region_Relative', 'Score_Source_3'
        ]

        for column in columns_to_clean:
            data[column] = pd.to_numeric(data[column], errors='coerce')

        return data

    def map_occupation(self, category):
        if pd.isna(category):
            return 'Unknown'
        elif category in ['Sales', 'Realty agents', 'Managers', 'Accountants', 'High skill tech', 'IT']:
            return 'Professional'
        elif category in ['Laborers', 'Core', 'Drivers', 'Cleaning', 'Low-skill Laborers']:
            return 'Skilled Labor'
        elif category in ['HR', 'Waiters/barmen', 'Cooking', 'Private service', 'Security', 'Secretaries']:
            return 'Service'
        elif category == 'Medicine':
            return 'Healthcare'
        else:
            return 'Other'

    def map_organization(self, category):
        if pd.isna(category):
            return 'Unknown'
        elif category == 'XNA':
            return 'Unknown'
        elif category in ['Self-employed', 'Government']:
            return 'Public Sector'
        elif category in ['Business Entity Type 3', 'Business Entity Type 2', 'Business Entity Type 1', 'Construction']:
            return 'Business'
        elif category in ['Trade: type 3', 'Trade: type 7', 'Trade: type 2', 'Agriculture']:
            return 'Trade'
        elif category in ['Military', 'Medicine', 'Housing', 'Industry: type 1', 'Industry: type 11', 'Bank', 'School', 'Industry: type 9', 'Postal', 'University']:
            return 'Institution'
        elif category in ['Transport: type 4', 'Transport: type 2', 'Transport: type 3', 'Transport: type 1']:
            return 'Transport'
        else:
            return 'Other'

    def clean_data(self, df):
        df.drop(columns = ['ID','Application_Process_Day','Application_Process_Hour', "Child_Count"], inplace=True)
        df = df.applymap(self.replace_unwanted_chars)
        df = self.clean_columns(df)
        df['Client_Gender'] = df['Client_Gender'].replace('XNA', 'Unknown')
        
        categorical_columns = ['Car_Owned','Bike_Owned','Active_Loan','House_Own','Mobile_Tag', 'Homephone_Tag', 'Workphone_Working','Default']
        for col in categorical_columns:
            df[col] = df[col].replace({1: 'yes', 0: 'no'})
        
        df['Client_Occupation'] = df['Client_Occupation'].apply(self.map_occupation)
        df['Type_Organization'] = df['Type_Organization'].apply(self.map_organization)

        # Convert 'Default' column from {'no', 'yes'} to {0, 1}
        df['Default'] = df['Default'].replace({'no': 0, 'yes': 1})
        
        return df

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own',
                                'Client_Gender', 'Client_Housing_Type', 'Client_Occupation', 'Type_Organization',
                                'Client_Marital_Status', 'Loan_Contract_Type', 'Homephone_Tag', 'Client_Contact_Work_Tag']
            numerical_cols = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 
                              'Employed_Days', 'Registration_Days', 'ID_Days', 'Population_Region_Relative',
                              'Credit_Bureau']

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
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info('Pipeline Completed')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info('Exception occurred in the get_data_transformation_object method')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            
            logging.info(f'Initial train_df head:\n{train_df.head()}')
            logging.info(f'Initial test_df head:\n{test_df.head()}')

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            logging.info(f'Cleaned train_df head:\n{train_df.head()}')
            logging.info(f'Cleaned test_df head:\n{test_df.head()}')

            logging.info('Data cleaning completed')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Default'  # Replace with your actual target variable
            drop_columns = [target_column_name, 'Own_House_Age', 'Score_Source_1', 'Social_Circle_Default', 'Score_Source_3']  # Adjust according to your dataset

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
