import os, sys
from ffp.logger import logging
from ffp.exception import ffpException
from datetime import datetime

FILE_NAME = "ffp.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME= "test.csv"
TARGET_COLUMN = "Price"

MODEL_FILE_NAME = "model.pkl"

Airline_TRANSFORMER_OBJECT_FILE_NAME = "Airline_transformer.pkl"
Source_Destination_TRANSFORMER_OBJECT_FILE_NAME = "Source_Destination_transformer.pkl"
Total_Stops_TRANSFORMER_OBJECT_FILE_NAME = "Total_Stops_transformer.pkl"
Additional_Info_TRANSFORMER_OBJECT_FILE_NAME= "Additional_Info_transformer.pkl"
reset_cols_TRANSFORMER_OBJECT_FILE_NAME= "reset_cols_transformer.pkl"


class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise ffpException(e, sys)


class DataIngestionCofig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = "flight_fare_prediction"
            self.collection_name = "ffp_data"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise ffpException(e, sys)
    
    #function to convert above details w.r.t. data ingestion class into dictionary, although not necessary
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise ffpException(e, sys)



class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
            self.missing_threshold:float = 0.2
            self.base_file_path = os.path.join("ffp_data.xlsx")         
        except Exception as e:
            raise ffpException(e, sys)



class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_transformation")

        self.transformed_train_path =  os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME)
        self.transformed_test_path =os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME)

        self.Airline_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Airline_TRANSFORMER_OBJECT_FILE_NAME)
        self.Source_Destination_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
        self.Total_Stops_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
        self.Additional_Info_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)
        self.reset_cols_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", reset_cols_TRANSFORMER_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")

        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        self.expected_error = 2000                                     ##determining threshold values for error and overfitting in a machine learning model involve a combination of experimentaion, analysis and domain knowledge
        self.overfitting_threshold = 200


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 200

class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        self.saved_model_dir = os.path.join("saved_models") 
        self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)


        self.Airline_pusher_transformer_path = os.path.join(self.model_pusher_dir, "transformer", Airline_TRANSFORMER_OBJECT_FILE_NAME)
        self.Source_Destination_pusher_transformer_path = os.path.join(self.model_pusher_dir,"transformer", Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
        self.Total_Stops_pusher_transformer_path = os.path.join(self.model_pusher_dir,"transformer", Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
        self.Additional_Info_pusher_transformer_path = os.path.join(self.model_pusher_dir,"transformer", Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)
        self.reset_cols_pusher_transformer_object_path = os.path.join(self.model_pusher_dir,"transformer", reset_cols_TRANSFORMER_OBJECT_FILE_NAME)
