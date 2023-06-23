from ffp.entity import artifact_entity, config_entity
from ffp.exception import ffpException
from ffp.logger import logging
from typing import Optional, Dict
import os,sys 
import pandas as pd
from ffp import utils, config
import numpy as np
from ffp.entity.config_entity import TARGET_COLUMN


class DataTransformation:
    def __init__(self,
                    data_transformation_config: config_entity.DataTransformationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact= data_ingestion_artifact
            self.Total_Stops_Dict = {'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}
            self.reset_cols = ['date', 'month', 'Dep_Time_hour', 'Dep_Time_min', 'Arrival_Time_hour','Arrival_Time_min', 
                               'Duration hour','Duration min','Airline','Source', 'Destination','Total_Stops', 'Additional_Info', 'Price']
        except Exception as e:
            raise ffpException(e,sys)

    def drop_missing_values(self, df)->pd.DataFrame:
        try:
            missing_value_count = df.isnull().values.sum()

            if missing_value_count>0:
                df.dropna(axis=0, inplace = True)
                df.reset_index(drop=True, inplace = True)
                logging.info(f"{missing_value_count} missing values are remove from the dataset.")
                return df
            
            logging.info(f"there are no missing values in the dataset")
            return df
        
        except Exception as e:
            raise ffpException(e,sys)

    def drop_duplicate_rows(self, df)->pd.DataFrame:
        try:
            duplicate_rows_count = len(df[df.duplicated()])     #df.duplictaed().sum()

            if duplicate_rows_count>0:
                df.drop_duplicates(keep= 'first', inplace = True)
                df.reset_index(drop = True, inplace = True)
                logging.info(f"{duplicate_rows_count} number of duplicate rows are removed from the dataset.")
                return df

            logging.info(f"there are no duplicate rows in teh dataset")
            return df       
        except Exception as e:
            raise ffpException(e,sys)

    def feature_encoding(self, df, column_name)->Dict:
        try:
            if column_name in ['Source','Destination']:  #Source and Destination should have same encoding for the cities name
                temp_list = list(df["Source"].unique())
                for x in df['Destination'].unique():
                    temp_list.append(x)
                temp_list = list(set(temp_list))
                logging.info(f"{column_name} has dictionary: {dict(zip(temp_list, range(len(temp_list))))}")
                return dict(zip(temp_list, range(len(temp_list)))) 
                
            else:
                temp_list = list(df[column_name].unique())
                logging.info(f"{column_name} has dictionary: {dict(zip(temp_list, range(len(temp_list))))}")
                return dict(zip(temp_list, range(len(temp_list))))
        except Exception as e:
            raise ffpException(e, sys)

    def remove_outliers(self, df, column_name, threshold_value)->pd.DataFrame:

        try:
            df[column_name]=np.where(df[column_name]>=threshold_value, df[column_name].median(), df[column_name])  
            df[column_name] = df[column_name].astype('int64')
            logging.info(f"Outlier removed from the {column_name} feature.")
            return df
        except Exception as e:
            raise ffpException(e, sys)
    
    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            logging.info("-----Transforming Train dataset-----")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            train_df['Destination'] = train_df['Destination'].replace("New Delhi", "Delhi")
            train_df['Additional_Info'] = train_df['Additional_Info'].replace('No Info', 'No info')


            train_df = self.drop_missing_values(df = train_df)
            train_df = self.drop_duplicate_rows(df = train_df)
            train_df = utils.sep_date_feature(train_df, "Date_of_Journey")
            train_df = utils.sep_time_feature(train_df, "Dep_Time")
            train_df = utils.sep_time_feature(train_df, "Arrival_Time")
            train_df = utils.duration_feature(train_df, "Duration")

            if train_df[train_df['Duration hour']==0].index != 0: 
                train_df.drop(train_df[train_df['Duration hour']==0].index, axis=0, inplace =True)

            Airline_Dict = self.feature_encoding(df= train_df, column_name= "Airline")
            train_df['Airline']= train_df['Airline'].map(Airline_Dict)

            Source_Destination_Dict = self.feature_encoding(df = train_df, column_name= 'Source')

            train_df['Source'] = train_df['Source'].map(Source_Destination_Dict)
            train_df['Destination'] = train_df['Destination'].map(Source_Destination_Dict)
            train_df['Total_Stops'] = train_df['Total_Stops'].map(self.Total_Stops_Dict)

            Additional_Info_Dict = self.feature_encoding(df=train_df, column_name='Additional_Info')
            train_df['Additional_Info'] = train_df['Additional_Info'].map(Additional_Info_Dict)

            train_df = self.remove_outliers(df=train_df, column_name='Price', threshold_value=30000)

            train_df.drop(['Route', 'year'], axis =1, inplace = True)      
            train_df = train_df.reindex(self.reset_cols, axis =1)

            logging.info("-----Transforming Test dataset-----")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)   
            test_df['Destination'] = test_df['Destination'].replace("New Delhi", "Delhi")
            test_df['Additional_Info'] = test_df['Additional_Info'].replace('No Info', 'No info')

            test_df = self.drop_missing_values(df=test_df)
            test_df = self.drop_duplicate_rows(df=test_df)
            test_df = utils.sep_date_feature(test_df, 'Date_of_Journey')
            test_df = utils.sep_time_feature(test_df, 'Dep_Time')
            test_df = utils.sep_time_feature(test_df, 'Arrival_Time')
            test_df = utils.duration_feature(test_df, 'Duration')

            if test_df[test_df['Duration hour']==0].index != 0: 
                test_df.drop(test_df[test_df['Duration hour']==0].index, axis=0, inplace =True)

            test_df['Airline'] = test_df['Airline'].map(Airline_Dict)
            test_df['Source'] = test_df['Source'].map(Source_Destination_Dict)
            test_df['Destination'] = test_df['Destination'].map(Source_Destination_Dict)
            test_df['Total_Stops'] = test_df['Total_Stops'].map(self.Total_Stops_Dict)
            test_df['Additional_Info'] = test_df['Additional_Info'].map(Additional_Info_Dict)

            test_df.drop(['Route', 'year'], axis =1, inplace = True)
            test_df = test_df.reindex(self.reset_cols, axis =1)

            utils.save_data(file_path= self.data_transformation_config.transformed_train_path , df = train_df)
            utils.save_data(file_path= self.data_transformation_config.transformed_test_path , df= test_df)

            utils.save_object(file_path=self.data_transformation_config.Airline_transformer_object_path, obj=Airline_Dict)
            utils.save_object(file_path=self.data_transformation_config.Source_Destination_transformer_object_path, obj=Source_Destination_Dict)
            utils.save_object(file_path=self.data_transformation_config.Total_Stops_transformer_object_path, obj=self.Total_Stops_Dict)
            utils.save_object(file_path=self.data_transformation_config.Additional_Info_transformer_object_path, obj=Additional_Info_Dict)
            utils.save_object(file_path = self.data_transformation_config.reset_cols_transformer_object_path, obj = self.reset_cols)     

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(                    
                    transformed_train_path = self.data_transformation_config.transformed_train_path,
                    transformed_test_path = self.data_transformation_config.transformed_test_path,
                    Airline_transformer_object_path = self.data_transformation_config.Airline_transformer_object_path,
                    Source_Destination_transformer_object_path = self.data_transformation_config.Source_Destination_transformer_object_path,
                    Total_Stops_transformer_object_path = self.data_transformation_config.Total_Stops_transformer_object_path,
                    Additional_Info_transformer_object_path = self.data_transformation_config.Additional_Info_transformer_object_path,
                    reset_cols_transformer_object_path = self.data_transformation_config.reset_cols_transformer_object_path
                )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}\n")
            return data_transformation_artifact
        
        except Exception as e:
            raise ffpException(e, sys)

