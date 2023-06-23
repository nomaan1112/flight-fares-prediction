 # Flight_Fare_Prediction
 ***********************
 **Flight Fare Prediction**

**Problem Statement:**

Air travel has become an essential aspect of modern lifestyles, with an increasing number of individuals seeking faster transportation options. Flight ticket prices fluctuate periodically, influenced by factors such as flight schedules, destinations, and durations, as well as special events like holidays or festive seasons. Consequently, acquiring a preliminary understanding of flight fares prior to trip planning can greatly benefit individuals in terms of cost and time savings.

**Goal:**
 
 The main goal is to predict the fares of the flights based on different factors available in the provided dataset.

 **Approach**

 The traditional machine learning workflow involves essential tasks such as data exploration, data cleaning, feature engineering, model building, and model testing. It is crucial to experiment with various machine learning algorithms to identify the most suitable approach for the given scenario.

 **Featured detail:**

* Airline: The name of the airline
* Date_of_Journey: The date of the journey
* Source: The source from which the service begins.
* Destination: The destination where the service ends.
* Route: The route taken by the flight to reach the destination.
* Dep_Time: The time when the journey starts from the source.
* Arrival_Time: Time of arrival at the destination.
* Duration: Total duration of the flight.
* Total_Stops: Total stops between the source and destination.
* Additional_Info: Additional information about the flight
* Price: The price of the ticket

 ## Modelling and Deployment process

 **Data Gathering**

The data for the current project is taken from Kaggle dataset, the link to the data is: 
https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh

**Data Description**

There are about 10k+ records of flight information such as airlines, data of journey, source, 
destination, departure time, arrival time, duration, total stops, additional information, and 
price. 

**Tool Used:**

* Python 3.8 is used while creating the environment. 
* VS Code is used as IDE.
* AWS is used for deployment
* HTML is used for developing the webpage for the instance prediction.
* GitHub is used as code repository.

**Data Pre-processing:**
 
* Initiating the pr-processing by removing the missing rows from the data.
* Removing the duplicate records from the data
* Splitting date, time and duration features and converting them into integers.
* Encoding the categorical data into integers using respective dictionaries
* Saving the dictionaries to encode the input values during prediction.

**Model trainer**

Pre-processed data has been passed to various machine learning regression models along with their hyper-parameters provided by GridSearchCV. The RandomForest Regressor outperform among the given regression algorithms.

**Model Evaluation:**

Currently trained model is evaluated with the previously saved trained model, if available. If currently trained model is better than the previously trained model then the current model and transformer objects are pushed and saved for the future use

**Prediction:**

Both the instance and batch prediction can be performed using the code. The app.py file can be used for instance prediction by taking input values from the user through HTML page and the main.py file can be used batch prediction or training the model as per the requirement.

**Deployment:**

AWS’s EC2 and ECR are used to deploy the instance prediction model.

# Thanks!!

 