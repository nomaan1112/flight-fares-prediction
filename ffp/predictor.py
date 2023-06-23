from ffp.exception import ffpException
import os, sys
from ffp.entity import artifact_entity, config_entity
from typing import Optional


class ModelResolver:                                              ##this class have function which will give the location of the folder inside "saved_models" folder, which have latest models/objects stored which are being used in teh production so that we can load them, and use them to do predictions and compare their accuracy with accuracy we get from the new model created by our sourcecode,,,,,,also this class have functions that will provide location where we will save the latest models/objects{i.e. those models/objects we want to save as they are better than models/objects which are currently being used in the production} 

    def __init__(self, model_registry ="saved_models",
                transformer_dir_name="transformer",
                model_dir_name = "model"):
        self.model_registry=model_registry
        os.makedirs(self.model_registry,exist_ok=True)
        self.transformer_dir_name = transformer_dir_name
        self.model_dir_name=model_dir_name

    
    
    def get_latest_dir_path(self)->Optional[str]:     ##this function will get the path/locatopn/folder of the latest model, i.e., the model currently being used in production from the folder{"saved_folders"} where all the models are saved, but if there is nothing inside the folder, then this function will return nothing
        try:
            dir_names = os.listdir(self.model_registry)  ##using this we will list all the content inside the model_registry i.e., the "saved_models" 
            if len(dir_names)==0:                        ##if there is nothing in the list then do nothing
                return None
            dir_names = list(map(int,dir_names))         ##if there is something then this code will map all the dir_names that we have got from above code with integer values, and finally return the folder with latest model from among the other folders present inside the "model_registry" folder
            latest_dir_name = max(dir_names)          ##as now the folder names are in integer form, the maximum out of them can be chosen
            return os.path.join(self.model_registry,f"{latest_dir_name}")    ##with this line of code, this function will return location of the folder inside the "model_registry" folder, which contain the latest model
        except Exception as e:
            raise ffpException(e, sys)
    
    def get_latest_model_path(self):       ##if we get the folder from the "saved_models" folders by using the previous function, which contain the latest models/objects which are currently being used in the production then this function will provide the path where the .pkl file which contain the model is stored,,similarly below function will provide path where object which is used to do the transformation of the data is stored
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir,self.model_dir_name,config_entity.MODEL_FILE_NAME)
        except Exception as e:
            raise ffpException(e, sys)

    def get_latest_transformer_path(self, transformer_name):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            return os.path.join(latest_dir,self.transformer_dir_name, transformer_name)
        except Exception as e:
            raise ffpException(e, sys)


    ##now function that will provide path where the models/objects will be stored which are found to be better than the models/objects which are currently being used in production
    def get_latest_save_dir_path(self)->str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry,f"{0}")                     ##if found none latest directory{then the trained model we created just now is the first model w.r.t the dataset}, then we will form one inside the "save_models" folder with number/name as '0', in which we will store the model/objects trained by our sourcecode
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))      ##if latest directory is found then this basename is used to get the latest directory's number, so that we can form a new one where we will store the model/objects which are better than the models/objects previously in use 
            return os.path.join(self.model_registry,f"{latest_dir_num+1}")
        except Exception as e:
            raise ffpException(e, sys)

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,config_entity.MODEL_FILE_NAME)
        except Exception as e:
            raise ffpException(e, sys)

    def get_latest_save_transformer_path(self, transformer_name):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name, transformer_name)
        except Exception as e:
            raise ffpException(e, sys)
