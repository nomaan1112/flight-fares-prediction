import pandas as pd
from ffp.exception import ffpException
from ffp.logger import logging
from ffp.config import mongo_client
import os, sys
import yaml
import dill

def get_collection_as_dataframe(database_name: str, collection_name: str)->pd.DataFrame():
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id")
            df = df.drop("_id", axis= 1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise ffpException(e, sys)

##function to save data/report in '.yaml' file{the format}
def write_yaml_file(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok = True)
        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise ffpException(e, sys)


def save_object(file_path:str, obj:object)->None:
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"{file_path.split('/')[-1]} object has been saved thorough utils")
    except Exception as e:
        raise ffpException(e, sys)

def load_object(file_path:str)->None:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
    
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ffpException(e, sys)

def save_data(file_path:str, df)->None:
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        df.to_csv(file_path, index= False, header =True)
    except Exception as e:
        raise ffpException(e, sys)
        
def load_data(file_path:str)->object:            ##although thi sfunction returning a dataframe, this function annotation is hinting tp return soem kind of object, not specifically a DataFrame, but if you want you can put "DataFrame " in place of "Object"
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The datarame {file_path} does not exist.")
        return pd.read_csv(file_path)
    except Exception as e:
        raise ffpException(e, sys)


# Functions for data transformation

def sep_date_feature(data:pd.DataFrame, col:str)-> pd.DataFrame:
    try:
        data['date'] = data[col].str.split("/").str[0]
        data['month'] = data[col].str.split("/").str[1]
        data['year'] = data[col].str.split("/").str[2]
    
        data['date'] = data['date'].astype(int)
        data['month'] = data['month'].astype(int)
        data['year'] = data['year'].astype(int)
    
        data.drop(col, axis=1, inplace= True)
        logging.info(f"'{col}' column has been splitted into 'date' and 'month' columns")
        return data
    except Exceptiin as e:
        raise ffpException(e,sys)

def sep_time_feature(data:pd.DataFrame,col:str)->pd.DataFrame:
    try:
        data[col]= data[col].str.split(" ").str[0]

        data[col+"_hour"]= data[col].str.split(":").str[0]
        data[col+"_min"] = data[col].str.split(":").str[1]

        data[col+"_hour"] = data[col+"_hour"].astype(int)
        data[col+"_min"] = data[col+"_min"].astype(int)

        data.drop(col, axis=1, inplace = True)
        logging.info(f"'{col}' column has been splitted into '{col+'_hour'}' and '{col+'_min'}' columns")
        return data
    except Exception as e:
        raise ffpException(e,sys)

def duration_feature(data:pd.DataFrame, col:str)->pd.DataFrame:
    try:
        duration = list(data[col])

        for i in range(len(duration)):
            if len(duration[i].split(' '))==2:
                pass
            else:
                if 'h' in duration[i]:                  # Check if duration contains only hours
                    duration[i] = duration[i] + ' 0m'  # Adds 0 minutes
                else:
                    duration[i] ='0h '+ duration[i]    # Adds 0 hours, if only minutes available
        data[col] = duration 

        data[col+' hour'] = data[col].str.split(" ").str[0].str.replace('h','')
        data[col+' min'] = data[col].str.split(" ").str[1].str.replace('m','')
        data[col+' hour'] = data[col+' hour'].astype(int)
        data[col+' min'] = data[col+' min'].astype(int)
        data.drop(col, axis=1, inplace = True)
        logging.info(f"'{col}' column has been splitted into '{col+'_hour'}' and '{col+'_min'}' columns")
        return data
    except Exception as e:
        raise ffpException(e,sys)
            



