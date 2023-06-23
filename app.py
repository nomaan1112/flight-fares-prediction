from flask import Flask, request, jsonify, url_for, render_template
from flask_cors import CORS, cross_origin
from ffp.predictor import ModelResolver
from ffp.exception import ffpException
from ffp.logger import logging
from ffp.entity import config_entity
from ffp.pipeline.training_pipeline import start_training_pipeline
from ffp.pipeline.batch_prediction import start_batch_prediction
from ffp import utils
import os,sys
import pandas as pd
import numpy as np
import datetime as dt
import pickle

app = Flask(__name__)

logging.info(f"Loading latest model from saved_models")

latest_dir = ModelResolver().get_latest_dir_path()
flight_model = utils.load_object(file_path=os.path.join(latest_dir,"model",config_entity.MODEL_FILE_NAME))

logging.info(f"Loading Latest trasnformers from saved_models folder!!")

Airline_path = os.path.join(ModelResolver().get_latest_dir_path(),"transformer",config_entity.Airline_TRANSFORMER_OBJECT_FILE_NAME)
Source_Destination_path = os.path.join(ModelResolver().get_latest_dir_path(),"transformer",config_entity.Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
Total_Stops_path = os.path.join(ModelResolver().get_latest_dir_path(),"transformer",config_entity.Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
Additional_Info_path = os.path.join(ModelResolver().get_latest_dir_path(),"transformer",config_entity.Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)
#reset_cols_path = os.path.join(latest_dir,"transformer",config_entity.reset_cols_TRANSFORMER_OBJECT_FILE_NAME)


Airline_transformer = utils.load_object(Airline_path)
Source_Destination_transformer = utils.load_object(Source_Destination_path)
Total_Stops_transformer = utils.load_object(Total_Stops_path)
Additional_Info_transformer = utils.load_object(Additional_Info_path)
#reset_cols_transformer = utils.load_object(reset_cols_path)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict_api', methods= ['POST'])
@cross_origin()
def predict_api():
    if request.method =='POST':
        try:
            data = [x for x in request.form.values()]                          ##extract all values from teh submitted forma and stores them in a list called "data"
            depart_time = data[0].split('T')[1]                                ##this extract time of departure from first element of the 'data' list, teh in ISO format{{example, the ISO format for May 9th, 2023 at 3:30 pm would be: "2023-05-09T15:30"}}, so the 'split' method is used to separate the date and time, and them the '[1]' index to extract only teh time
            result = dt.datetime.strptime(data[1], '%H:%M')-dt.datetime.strptime(depart_time,'%H:%M')     ##this calculates teh difference between teh scheduled arrival time and teh departure time. it first converts teh scheduled arrival time (which si teh second element of teh 'data' list) and teh departure time(which was extractd previously) to 'datetime' objects using teh 'strptime' method, and then subtracts them. the result is a 'timedelta' object.
            print(result.seconds/60)                                           ##the line prints the duration of the flight in minutes. teh 'seconds' attribute of teh 'timedelta' object is used to get the duration in seconds, which i then divided by 60 to get the duration in minutes.

            if dt.datetime.strptime(depart_time,'%H:%M')==dt.datetime.strptime(data[1], '%H:%M'):    ##checks, if departre time an darrival time same then it sets duration hours to 24 hours but if not then from teh diference is converted into hours and minutes
                duration_h =24
                duration_min = 0
            else:
                duration_h = (result.seconds/60)//60                    ##double forward slash calculate the integer number of hours 
                duration_min = (result.seconds/60)%60                   ##the modulus operatoru '%' calculaytes the remaining minutes
            print(duration_h, duration_min)

            filtered_data = []

            filtered_data.append(data[0].split('T')[0].split('-')[2])   ##departure 'date'
            filtered_data.append(data[0].split('T')[0].split('-')[1])   ##departure 'month'
            filtered_data.append(data[0].split('T')[1].split(':')[0])   ##'Dep_Time_hour'
            filtered_data.append(data[0].split('T')[1].split(':')[1])   ##'Dep_Time_min'
            filtered_data.append(data[1].split(':')[0])                  ##'Arrival_Time_hour'
            filtered_data.append(data[1].split(':')[1])                  ##'Arrival_Time_min'
            filtered_data.append(duration_h)                             ##'Duration hour'
            filtered_data.append(duration_min)                           ##'Duration min'
            filtered_data.append(data[2])                                ##'Airline'
            filtered_data.append(data[3])                                ##'Source'
            filtered_data.append(data[4])                                ##'Destination'
            filtered_data.append(data[5])                                ##'Total_Stops'
            filtered_data.append(data[6])                                ##'Additional_Info'
            print(filtered_data)

            filtered_data[8] = int((pd.Series(filtered_data[8]).map(Airline_transformer)).values)                  ##to apply map() function on list 'filtered_data' elements, we need to convert those elements into Pandas Series using 'pd.Series()', as 'map()' function in panadas is a method of the Series object, not a built-in function.
            filtered_data[9] = int((pd.Series(filtered_data[9]).map(Source_Destination_transformer)).values)       ##here we have replaced the list 'filtered_data' elemets at 8,9,10,11,12 index position with their mapped version present in the dictionaries i have imported from config_entity using utils
            filtered_data[10] = int((pd.Series(filtered_data[10]).map(Source_Destination_transformer)).values)     ##suppose a element in the list is "abc" then pd.Series("abc") will form a series with one element "0    abc" then mpping it "pd.Series("abc").map({"abc":2})" will give a series with one element "0    2", now extract this element '2' using "(pd.Series("abc").map({"abc":2})).values" and converting it into integer form using "int((pd.Series("abc").map({"abc":2})).values)"
            filtered_data[11] = int((pd.Series(filtered_data[11]).map(Total_Stops_transformer)).values)            ##in above mentoned way will map elements in the list
            filtered_data[12] = int((pd.Series(filtered_data[12]).map(Additional_Info_transformer)).values)

            print(filtered_data)
    
            filtered_data = [int(x) for x in filtered_data]            ##converting remaining values in the list into integer
            final_input= np.array(filtered_data).reshape(1,-1)         ##l = [0,1,2],,,np.array(l) will create 1D NumPy array: array(['0','1','2']), then 'reshape(1,-1)' is used to convert the 1D array into a 2D array with one row and an unknon number of columns: array([['0','1','2']]), thsi 2D array can eb used as input to machine learning models that expect a 2D array as input
            print(final_input)

            output = flight_model.predict(final_input)[0]
            print(output)

            return render_template('index.html', output_text="The Price of the fight is {}.".format(round(output,2)))

        except Exception as e:
            raise ffpException(e,sys)
    
    else:
        return render_template('index.html')

if __name__=="__main__":
    try:
        #app.run(debug=True)                ##in some cases, depending on our operating system an dnetwork configuration, 'debug = True' may prevent external clienets from connecting to teh server , in which case using 'host="0.0.0.0" instead may resolve the issue.
        #app.run(host="0.0.0.0")
        app.run(host="0.0.0.0", port=8080)

    except Exception as e:
       raise ffpException(e, sys)


"""
NOTE: As this flask app is running on a remote server or virtual machine provided by my coaching website. so to access this flask app , i need to use the URL of that 
server or virtual machine where this app is running. 
As i have built this app on a VS Code instance which is hosted on teh website `https://white-musician-uflwx.ineuron.app/?folder=/config/workspace`. This means my app is 
running on that VS Code instance, whic is not  my local machine. So i need to access my app using the URL of that VS Code instance.
Hence i used `https://white-musician-uflwx.ineuron.app:5000` to run teh app ceated by me after running the app.py file. This URL will tell my web browser to access teh 
Flask app running on teh VS Code instance using port 5000, which is default port used by Flask apps.  
Summary: the `http://localhost:5000` didn't work beaacuse my flask app is not running on my local machine, but on a remote server or virtula machine provided by ineuron. 
To acces teh app, i need to use the URL of that renote sever or virtul machine , and add `:5000` at teh end to specify teh port number.

Also note that when we run flask app , it start a web server that listens for incoming requests on a specific host and port. By default teh Flask development server 
listens only on teh localhost interface(127.0.0.1) and is not accessible from other machines. So if we want to access the flask app from other machine or from a remote server,
we need to set the host parameter tp "0.0.0.0" so that it listens on all available network interfaces. This makes our flaks app accessible to any machine that can reach the network address of teh machine running the flaks app.
Hence i used `app.run(host="0.0.0.0")`, this makes my flask app listes om all availabel network interfaces, which allows it to be accessed remotely.
"""