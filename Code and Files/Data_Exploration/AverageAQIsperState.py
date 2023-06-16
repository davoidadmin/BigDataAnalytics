from pyspark.sql import *
from pyspark.sql.functions import avg
from pyspark.sql import functions as F
import time
import pandas as pd
import matplotlib.pyplot as plt



start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Average AQIs per State") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

df = spark.read.load("/home/giuseppe/Second_Project/Preprocessed_Dataset.csv" , format="csv", sep=",", header="true")
df=df.select((F.to_date("Date Local","yyyy-MM-dd").alias("Date")),"State","NO2 AQI","O3 AQI","SO2 AQI","CO AQI")
df=df.groupBy("State").agg(F.avg(F.col("NO2 AQI")).alias("NO2"),F.avg(F.col("O3 AQI")).alias("O3"),F.avg(F.col("SO2 AQI")).alias("SO2"),F.avg(F.col("CO AQI")).alias("CO"))

def AverageAQIsperState(pollutant,fig_number,color,dataframe):
 df=dataframe.orderBy(pollutant, ascending=False)
 df_pandas=df.toPandas()
 
 df_pandas.plot(x='State',y=pollutant,kind='bar',legend=None,color=color)
 plt.title(pollutant+' average AQIs per state', fontdict=None, loc='center', pad=None)
 plt.ylabel('Average AQI')
 plt.figure(fig_number)
 
AverageAQIsperState("NO2",1,"blue",df)
AverageAQIsperState("O3",2,"orange",df)
AverageAQIsperState("SO2",3,"green",df)
AverageAQIsperState("CO",4,"red",df)

plt.show()

end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
