from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import substring
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax



start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Decision Tree Regressor") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

df=spark.read.load("/home/giuseppe/Second_Project/Preprocessed_Dataset.csv" , format="csv", sep=",", header="true")
df=df.select("State","County","Site Num","Day","Month","Year",(F.to_date("Date Local","yyyy-MM-dd").alias("Date")),"NO2 AQI","O3 AQI","SO2 AQI","CO AQI")
df=df.orderBy("Date")
df=df.groupBy("Date").agg(avg("NO2 AQI").alias("NO2"),avg("O3 AQI").alias("O3"),avg("SO2 AQI").alias("SO2"),avg("CO AQI").alias("CO"))

def SARIMAX(pollutant,fig_number1,fig_number2,color,dataframe):
 df=dataframe.select("Date",pollutant)
 df=df.toPandas()
 df.index = pd.to_datetime(df["Date"])

 train = df[:'2014-12-31']
 test = df['2015-01-01':'2016-12-31']
 
 # Fourier terms
 fourier = pd.DataFrame(index=df.index)
 # Frequency is being set to 365.25 because we have leap years
 fourier['sin_1'] = np.sin(2 * np.pi * fourier.index.dayofyear / 365.25)
 fourier['cos_1'] = np.cos(2 * np.pi * fourier.index.dayofyear / 365.25)
 fourier['sin_2'] = np.sin(4 * np.pi * fourier.index.dayofyear / 365.25)
 fourier['cos_2'] = np.cos(4 * np.pi * fourier.index.dayofyear / 365.25)
 fourier_train = fourier.iloc[:len(train), :]
 fourier_test = fourier.iloc[len(train):(len(train)+len(test)), :]

 arima = pm.auto_arima(train[pollutant],exogenous=fourier_train, start_p=1, start_q=0,stepwise=True,suppress_warnings=True,error_action='ignore')
 
 predictions = arima.predict(n_periods=len(test), exogenous=fourier_test)
 sarima_pred = pd.DataFrame(np.c_[predictions],index=test.index)
 rmse = np.sqrt(mean_squared_error(test[pollutant], sarima_pred))
 mae = mean_absolute_error(test[pollutant], sarima_pred)

 plt.figure(fig_number1)
 plt.scatter(test[pollutant],sarima_pred,color=color)
 plt.xlabel("Real "+pollutant+" values")
 plt.ylabel(pollutant+" prediction")
 plt.title('Predictions for '+ pollutant+' with SARIMAX\nRoot Mean Square Error: '+str(round(rmse,2))+'\nMean Absolute Error: '+str(round(mae,2)))

 plt.figure(fig_number2)
 plt.plot(train["Date"],train[pollutant],color="black",label="Train data")
 plt.plot(test["Date"],test[pollutant],color="darkgray",label="Test data")
 plt.plot(sarima_pred, color=color,label="Predicted data")
 plt.legend(loc='upper left', ncol=1, fancybox=True, shadow=True)
 plt.xlabel("Date")
 plt.ylabel(pollutant+" AQI")
 plt.title('Temporal Forecast for '+pollutant+' with SARIMAX')


SARIMAX("NO2",1,2,"blue",df)
SARIMAX("O3",3,4,"orange",df)
SARIMAX("SO2",5,6,"green",df)
SARIMAX("CO",7,8,"red",df)
 
plt.show()

end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
