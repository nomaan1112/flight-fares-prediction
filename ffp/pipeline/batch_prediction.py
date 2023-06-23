from ffp.exception import ffpException
from ffp.logger import logging
from ffp.predictor import ModelResolver
from ffp import utils
import os, sys
import pandas as pd 
import numpy as np 
from datetime import datetime
from scipy.stats import ks_2samp
from ffp.entity.config_entity import TARGET_COLUMN

from ffp.utils import load_object
from ffp.components.data_ingestion import DataIngestion
from ffp.components.data_transformation import DataTransformation
from ffp.entity import artifact_entity, config_entity

PREDICTION_DIR = "Prediction_Output"

Airline_TRANSFORMER_OBJECT_FILE_NAME = "Airline_transformer.pkl"
Source_Destination_TRANSFORMER_OBJECT_FILE_NAME = "Source_Destination_transformer.pkl"
Total_Stops_TRANSFORMER_OBJECT_FILE_NAME = "Total_Stops_transformer.pkl"
Additional_Info_TRANSFORMER_OBJECT_FILE_NAME= "Additional_Info_transformer.pkl"
reset_cols_TRANSFORMER_OBJECT_FILE_NAME= "reset_cols_transformer.pkl"

def start_batch_prediction(input_file_path):
    try:
        logging.info(f"Reading input file :{input_file_path}")
        prediction_df = pd.read_excel(input_file_path)
        print(prediction_df.shape)
        prediction_df.replace({"na":np.NAN}, inplace = True)

        
        
        ## Now first we will check the validity of the input_file

        #so first we will load the latest dataset "ffp.csv" from the artifact folder, as this will be the base dataset using which the current model was built, that we will be using to form the prediction file
        logging.info(f"Reading base dataframe")
        base_file_dir = os.path.join(os.getcwd(),"artifact")
        dir_names = os.listdir(base_file_dir) 
        dir_names.sort() 
        latest_dir_name = dir_names[len(dir_names)-1]
        base_file_path = os.path.join(base_file_dir,f"{latest_dir_name}","data_ingestion/feature_store/ffp.csv")
        logging.info(f"Reading base file :{base_file_path}")    
        base_df = pd.read_csv(base_file_path)
        base_df.drop(TARGET_COLUMN, axis = 1, inplace = True)                             ##as the input file does not have the target column, so no need to include it here in teh base_df while doing teh validation
        base_df.replace({"na":np.NAN}, inplace = True)
        logging.info(f"Replace na value in base df")
        
        #drop the columns with null values more than 20 percent
        logging.info(f"Drop null values columns from base_df")
        base_df_null_report = base_df.isna().sum()/base_df.shape[0]
        base_df_drop_column_names = base_df_null_report[base_df_null_report>0.2].index
        base_df.drop(list(base_df_drop_column_names),axis=1, inplace = True)
        #input file
        logging.info(f"Drop null values colums from input df")
        null_report = prediction_df.isna().sum()/prediction_df.shape[0]
        drop_column_names = null_report[null_report>0.2].index
        prediction_df.drop(list(drop_column_names),axis = 1, inplace = True)
        #raise exception if no columns left
        if len(prediction_df.columns)==0:
            logging.info(f"no column left after removing those with more than 20 percent null values")
            raise Exception("no column left after removing those with more than 20 percent null values")
            
        #else we do further validation
        #now will validate the presenec of required number of columns
        logging.info(f"checking the presence of required number of columns")    
        base_columns = base_df.columns
        current_columns = prediction_df.columns

        missing_columns = []
        for base_column in base_columns:
            if base_column not in current_columns:
                missing_columns.append(base_column)
        if len(missing_columns)>0:
            raise Exception("required number of columns are not present")
        
        #otherwise carry on with the data validation operations
        logging.info(f"now comparing datatypes")
        base_dtypes = base_df.dtypes
        current_dtypes = prediction_df.dtypes
        #current_columns = prediction_df.columns

        invalid_datatype_columns = []
        for i, datatype in enumerate(base_dtypes):
            if datatype != current_dtypes[i]:
                invalid_datatype_columns.append(current_columns[i])

        if len(invalid_datatype_columns)>0:
            raise Exception(f"{invalid_datatype_columns} columns don't have data types as per in the base dataset.")

        logging.info(f"as all columns are available in input file hence now checking if there is data drift in it")
        base_num_cols= [cols for cols in base_columns if base_df[cols].dtype !='O']
        current_num_cols = [cols for cols in current_columns if prediction_df[cols].dtype !='O']
        if len(base_num_cols)>0 and len(current_num_cols)>0:
                for base_column in base_num_cols:
                    base_data, current_data = base_df[base_column], prediction_df[base_column]
                    #Null hypothesis is that both column data drawn from same distrubtion
                
                    logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")  ##this logging is important because it will log the datatype of the both data whose distribution is to be tested, so that we can know when any error occur because of the datatype mismatch, like happened initially then we changed the datatype of the all the features except the 'class' feature/column to float so that no mismatch occur, as mostly while working on data, sometime the datatype of some of the values of a feature changes.
                    same_distribution =ks_2samp(base_data,current_data)

                    if same_distribution.pvalue>0.05:
                        #We are accepting null hypothesis, i.e. same distribution
                        logging.info(f"no data drift")
                    else:
                        raise Exception("data drift present")
                        #different distribution
        
                





        ##transformation

        prediction_df['Destination'] = prediction_df['Destination'].replace("New Delhi", "Delhi")
        prediction_df['Additional_Info'] = prediction_df['Additional_Info'].replace('No Info', 'No info')


        logging.info(f"Creating data_transformation object to use soem of its functions")
        data_transformation = DataTransformation(data_transformation_config = config_entity.DataTransformationConfig, data_ingestion_artifact = artifact_entity.DataIngestionArtifact)
        prediction_df = data_transformation.drop_missing_values(df=prediction_df)
        prediction_df = data_transformation.drop_duplicate_rows(df=prediction_df)
        
        prediction_df = utils.sep_date_feature(prediction_df, 'Date_of_Journey')
        prediction_df = utils.sep_time_feature(prediction_df, 'Dep_Time')
        prediction_df = utils.sep_time_feature(prediction_df, 'Arrival_Time')
        prediction_df = utils.duration_feature(prediction_df, 'Duration')

        if prediction_df[prediction_df['Duration hour']==0].index != 0: 
                prediction_df.drop(prediction_df[prediction_df['Duration hour']==0].index, axis=0, inplace =True)

        #loading model & transformer objects
        logging.info(f"Creating model resolver object to load Latest model & transformer objects")
        model_resolver = ModelResolver(model_registry="saved_models")

        Airline_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Airline_TRANSFORMER_OBJECT_FILE_NAME))
        Source_Destination_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Source_Destination_TRANSFORMER_OBJECT_FILE_NAME))
        Total_Stops_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Total_Stops_TRANSFORMER_OBJECT_FILE_NAME))
        Additional_Info_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Additional_Info_TRANSFORMER_OBJECT_FILE_NAME))
        reset_cols_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(reset_cols_TRANSFORMER_OBJECT_FILE_NAME))


        prediction_df['Airline'] = prediction_df['Airline'].map(Airline_transformer)
        prediction_df['Source'] = prediction_df['Source'].map(Source_Destination_transformer)
        prediction_df['Destination'] = prediction_df['Destination'].map(Source_Destination_transformer)
        prediction_df['Total_Stops'] = prediction_df['Total_Stops'].map(Total_Stops_transformer)
        prediction_df['Additional_Info'] = prediction_df['Additional_Info'].map(Additional_Info_transformer)

        prediction_df.drop(['Route', 'year'], axis =1, inplace = True)      
        prediction_df = prediction_df.reindex(reset_cols_transformer, axis =1)

        prediction_df.drop(TARGET_COLUMN, axis =1, inplace =True)       ##as it got included because of the above reset_cols_transformer, as this transformer list has "Price" column/feature in it

        print(prediction_df.columns, "\n", prediction_df.shape)

        logging.info(f"Loading model to make prediction")
        model = utils.load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(prediction_df)
       
        prediction_df["Prediction"] = prediction
        prediction_df["Prediction"] = round(prediction_df["Prediction"],0)

        os.makedirs(PREDICTION_DIR, exist_ok=True)

        prediction_file_name = os.path.basename(input_file_path).replace(".xlsx",f"{datetime.now().strftime('%m-%d-%Y__%H:%M:%S')}.xlsx")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name )

        prediction_df.to_excel(prediction_file_path, index=False, header=True)
        return prediction_file_path

    except Exception as e:
        raise ffpException(e, sys)
