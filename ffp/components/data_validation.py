from ffp.entity import artifact_entity, config_entity
from ffp.exception import ffpException
from ffp.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os,sys 
import pandas as pd
from ffp import utils, config
import numpy as np
from ffp.entity.config_entity import TARGET_COLUMN

class DataValidation:

    def __init__(self, data_validation_config:config_entity.DataValidationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise ffpException(e, sys)
    
    def drop_missing_values_columns(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            logging.info(f"selecting column names which contain null value above {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            if len(list(drop_column_names))>0:
                logging.info(f"columns to drop: {list(drop_column_names)}")
                self.validation_error[report_key_name]= list(drop_column_names)
                df.drop(list(drop_column_names), axis = 1, inplace = True)
            else:
                logging.info(f"no columns to drop")

            #return None if no column left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise ffpException(e, sys)


    #after dropping some columns with missing values greater than threshold value, will check now if required number of columns exists or not
    def is_required_columns_exists(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available.]")
                    missing_columns.append(base_column)
            
            if len(missing_columns)>0:
                self.validation_error[report_key_name]= missing_columns
                return False
            return True
        except Exception as e:
            raise ffpException(e, sys)

    def compare_datatypes(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:
            base_dtypes = base_df.dtypes
            current_dtypes = current_df.dtypes
            current_columns = current_df.columns

            invalid_datatype_columns = []
            for i, datatype in enumerate(base_dtypes):
                if datatype != current_dtypes[i]:
                    invalid_datatype_columns.append(current_columns[i])

            if len(invalid_datatype_columns)>0:
                self.validation_error[report_key_name]= invalid_datatype_columns
                return False
                logging.info(f"{invalid_datatype_columns} columns don't have data types as per in the base dataset. ")

            logging.info(f"features have data types as per the base dataset.")
            return True    
            
        except Exception as e:
            raise ffpException(e, sys)

    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        try:
            drift_report = {}
            base_num_cols= [cols for cols in base_df.columns if base_df[cols].dtype !='O']
            current_num_cols = [cols for cols in current_df.columns if current_df[cols].dtype !='O']

            if len(base_num_cols)>0 and len(current_num_cols)>0:
                for base_column in base_num_cols:
                    base_data, current_data = base_df[base_column], current_df[base_column]
                    #Null hypothesis is that both column data drawn from same distrubtion
                
                    logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")  ##this logging is important because it will log the datatype of the both data whose distribution is to be tested, so that we can know when any error occur because of the datatype mismatch, like happened initially then we changed the datatype of the all the features except the 'class' feature/column to float so that no mismatch occur, as mostly while working on data, sometime the datatype of some of the values of a feature changes.
                    same_distribution =ks_2samp(base_data,current_data)

                    if same_distribution.pvalue>0.05:
                        #We are accepting null hypothesis, i.e. same distribution
                        drift_report[base_column]={
                            "pvalues":float(same_distribution.pvalue),
                            "same_distribution": True
                        }
                    else:
                        drift_report[base_column]={
                            "pvalues":float(same_distribution.pvalue),
                            "same_distribution":False
                        }
                        #different distribution

                self.validation_error[report_key_name]=drift_report
            
            else:
                logging.info(f"no numerical columms in to check for data drift")

        except Exception as e:
            raise ffpException(e, sys)

    
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_excel(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN}, inplace = True)
            logging.info(f"Replace na value in base df")
            #base_df has na as null
            logging.info(f"Drop null values columns from base_df")
            base_df = self.drop_missing_values_columns(df= base_df, report_key_name="missing_values_column_dropped_within_base_dataset")

            
            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Drop null values colums from train_df")
            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_columns_dropped_within_train_dataset")
            train_df_columns_status = self.is_required_columns_exists(base_df= base_df, current_df= train_df, report_key_name= "Missing columns within train dataset")

            if train_df_columns_status:
                logging.info(f"All features are available in the train dataset, processing with their data types.")
                self.compare_datatypes(base_df= base_df, current_df= train_df, report_key_name= "datatypes within train dataset")
                logging.info(f"checking data drift in the train_df")
                self.data_drift(base_df= base_df, current_df= train_df, report_key_name= "data_drift_within_train_dataset")       #as there is only one column with inetger datatype, hence to check data drift in that case

            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Drop null values colums from test_df")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="missing_values_columns_dropped_within_test_dataset")
            test_df_columns_status = self.is_required_columns_exists(base_df= base_df.iloc[:, base_df.columns!=TARGET_COLUMN], current_df= test_df, report_key_name= "Missing columns within test dataset")    ##from base_df, i excluded the "Price" column becuase it is not present in the test_df for obvious reasons

            if test_df_columns_status:
                logging.info(f"All features are available in the test dataset, processing with their data types.")
                self.compare_datatypes(base_df= base_df.iloc[:, base_df.columns!=TARGET_COLUMN], current_df= test_df, report_key_name= "datatypes within test dataset")
                logging.info(f"checking data drift in the test_df")
                self.data_drift(base_df= base_df.iloc[:, base_df.columns!=TARGET_COLUMN], current_df= test_df, report_key_name= "data_drift_within_test_dataset")

            #write validation report
            logging.info(f"writing validation report in yaml file")
            utils.write_yaml_file(file_path= self.data_validation_config.report_file_path, data= self.validation_error)

            #validation artifact
            data_validation_artifact= artifact_entity.DataValidationArtifact(self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
            
        except Exception as e:
            raise ffpException(e, sys)
