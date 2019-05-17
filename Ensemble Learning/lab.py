import pandas as pd
import numpy as np
import xgboost


data_day=pd.read_csv("Bike-Sharing-Dataset/day.csv",usecols=['season','holiday','weekday','workingday','weathersit','cnt'])
data_hour=pd.read_csv("Bike-Sharing-Dataset/hour.csv")
data_day.sample(frac=1).head()
training_data=data_day.iloc[:int(0.7*len(data_day))].reset_index(drop=True)
testing_data=data_day.iloc[int(0.7*len(data_day)):].reset_index(drop=True)


model10=xgboost.XGBRegressor()
model10.fit(training_data.iloc[:,:-1],training_data.iloc[:,-1:])
predict=model10.predict(testing_data.iloc[:,:-1])
