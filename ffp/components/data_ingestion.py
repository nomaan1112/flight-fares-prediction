from ffp import utils
from ffp.entity import config_entity
from ffp.entity import artifact_entity
from ffp.logger import logging
from ffp.exception import ffpException
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self, data_ingestion_config: config_entity.DataIngestionCofig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ffpException(e, sys)
    
    def initiate_data_ingestion(self)-> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name, 
                collection_name = self.data_ingestion_config.collection_name)
            
            
            logging.info(f"saving data in feature store folder")

            #replace na with Nan in the dataframe
            df.replace(to_replace='na', value = np.NAN, inplace = True)

            #save data in feature store folder
            logging.info("Create feature store directory/folder if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok = True)

            logging.info("Save the dataset to feature store folder")
            df.to_csv(path_or_buf = self.data_ingestion_config.feature_store_file_path, index = False, header = True)

            #split dataset into train and test set
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)

            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("save df to dataset folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise ffpException(e, sys)