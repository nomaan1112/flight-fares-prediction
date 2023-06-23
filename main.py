from ffp.pipeline.training_pipeline import start_training_pipeline
from ffp.pipeline.batch_prediction import start_batch_prediction
from ffp.exception import ffpException
import sys

file_path = "/config/workspace/input_files/input_file.xlsx"
print(__name__)
if __name__=="__main__":
     try:
          #start_training_pipeline()
          output_file = start_batch_prediction(input_file_path = file_path)
          print(output_file)
     except Exception as e:
          raise ffpException(e,sys)