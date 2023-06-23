from ffp.exception import ffpException
from ffp.logger import logging
from ffp.entity import config_entity, artifact_entity
from ffp.components.data_transformation import DataTransformation
from ffp.predictor import ModelResolver
from ffp import utils
import pandas as pd
import sys,os

from sklearn.metrics import mean_absolute_error
#from ffp.entity.config_entity import TARGET_COLUMN

class ModelEvaluation:
    def __init__(self, 
                    model_eval_config: config_entity.ModelEvaluationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                    data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                    model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*10} Model Evaluation {'<<'*10}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
            self.data_transformation = DataTransformation(data_transformation_config= config_entity.DataTransformationConfig, data_ingestion_artifact = self.data_ingestion_artifact)         ##initiated this because wanted to use the functions prsent in teh 'DataTransformation" class

        except Exception as e:
            raise ffpException(e, sys)
    
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("if 'saved_models' folder has model then we will compare which model is best btw currently trained model or" 
                "latest model available in the 'saved_models'")

            latest_dir_path = self.model_resolver.get_latest_dir_path()
            
            if latest_dir_path is None:
                logging.info(f"Previous model is not available.")
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_error=None)
                logging.info(f"Model Evaluation Artifact--- {model_eval_artifact}\n")
                return model_eval_artifact
            
            #getting paths of all the latest transforners, objects etc that are in production
            logging.info("Finding location of all the latest transformers")
            Airline_transformer_object_path = self.model_resolver.get_latest_transformer_path(transformer_name = config_entity.Airline_TRANSFORMER_OBJECT_FILE_NAME)
            Source_Destination_transformer_object_path = self.model_resolver.get_latest_transformer_path(transformer_name = config_entity.Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
            Total_Stops_transformer_object_path = self.model_resolver.get_latest_transformer_path(transformer_name = config_entity.Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
            Additional_Info_transformer_object_path = self.model_resolver.get_latest_transformer_path(transformer_name = config_entity.Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)
            reset_cols_transformer_object_path = self.model_resolver.get_latest_transformer_path(transformer_name = config_entity.reset_cols_TRANSFORMER_OBJECT_FILE_NAME)
            logging.info("Finding location of latest model_path")  
            model_path = self.model_resolver.get_latest_model_path()

            #loading those latest transformers and model objects that are in production
            logging.info("Previously trained transformers objects")
            previous_Airline_transformer= utils.load_object(file_path= Airline_transformer_object_path)
            previous_Source_Destination_transformer= utils.load_object(file_path= Source_Destination_transformer_object_path)
            previous_Total_Stops_transformer= utils.load_object(file_path= Total_Stops_transformer_object_path)
            previous_Additional_Info_transformer= utils.load_object(file_path= Additional_Info_transformer_object_path)
            previous_reset_cols_transformer= utils.load_object(file_path= reset_cols_transformer_object_path)
            logging.info("Previously trained model objects")
            previous_model= utils.load_object(file_path= model_path)

            print(type(previous_Airline_transformer))
            print(previous_Airline_transformer)
            print(type(previous_Source_Destination_transformer))
            print(previous_Source_Destination_transformer)
            print(type(previous_Total_Stops_transformer))
            print(previous_Total_Stops_transformer)
            print(type(previous_Additional_Info_transformer))
            print(previous_Additional_Info_transformer)
            print(type(previous_reset_cols_transformer))
            print(previous_reset_cols_transformer)
            print(type(previous_model))
            print(previous_model)

            #loading the test dataframe that we ingested using data_ingestion
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #now doing transformation to the nely ingested data using all the ways and transformers that were in use in production{currently there is nothing therefore we using the same ways that we used to transform the data whiel making this project}
            test_df['Destination'] = test_df['Destination'].replace("New Delhi", "Delhi")
            test_df['Additional_Info'] = test_df['Additional_Info'].replace('No Info', 'No info')

            test_df = self.data_transformation.drop_missing_values(df=test_df)
            test_df = self.data_transformation.drop_duplicate_rows(df=test_df)
            test_df = utils.sep_date_feature(test_df, 'Date_of_Journey')
            test_df = utils.sep_time_feature(test_df, 'Dep_Time')
            test_df = utils.sep_time_feature(test_df, 'Arrival_Time')
            test_df = utils.duration_feature(test_df, 'Duration')

            if test_df[test_df['Duration hour']==0].index != 0: 
                test_df.drop(test_df[test_df['Duration hour']==0].index, axis=0, inplace =True)

            test_df['Airline'] = test_df['Airline'].map(previous_Airline_transformer)
            test_df['Source'] = test_df['Source'].map(previous_Source_Destination_transformer)
            test_df['Destination'] = test_df['Destination'].map(previous_Source_Destination_transformer)
            test_df['Total_Stops'] = test_df['Total_Stops'].map(previous_Total_Stops_transformer)
            test_df['Additional_Info'] = test_df['Additional_Info'].map(previous_Additional_Info_transformer)

            test_df.drop(['Route', 'year'], axis =1, inplace = True)
            test_df = test_df.reindex(previous_reset_cols_transformer, axis =1)

            x_test, y_true = test_df.iloc[:,:-1], test_df.iloc[:,-1] 

            # Finding MAE using previously trained model
            y_pred = previous_model.predict(x_test)
            previous_model_error = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE using previously trained model: {previous_model_error}")

            #transformed test dataframe using currently formed transformers
            ttest_df = pd.read_csv(self.data_transformation_artifact.transformed_test_path)
            xt_test, yt_true = ttest_df.iloc[:,:-1], ttest_df.iloc[:,-1]

            #Currently trained model objects
            logging.info("Currently trained model objects")            
            current_model  = utils.load_object(file_path=self.model_trainer_artifact.model_path)

            yt_pred = current_model.predict(xt_test)
            current_model_error = mean_absolute_error(yt_true, yt_pred)
            logging.info(f"MAE using currently trained model: {current_model_error}")
            

            if current_model_error >= previous_model_error:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_error= previous_model_error - current_model_error)

            logging.info(f"Model eval artifact: {model_eval_artifact}\n")
            return model_eval_artifact

        except Exception as e:
            raise ffpException(e,sys)