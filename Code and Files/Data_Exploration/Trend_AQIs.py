from pyspark.sql import *
from pyspark.sql.functions import substring
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql import functions as F
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import pandas as pd
import matplotlib.pyplot as plt



start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Job1(2)") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

def plot_data(pollutant1,pollutant2,fig_number,ylimit,color1,color2):
 df = spark.read.load("/home/giuseppe/Second_Project/Preprocessed_Dataset.csv" , format="csv", sep=",", header="true")

 df=df.select((F.to_date("Date Local","yyyy-MM-dd").alias("Date")),pollutant1+" AQI",pollutant2+" AQI")
 df=df.orderBy("Date")
 df=df.groupBy("Date").agg(F.avg(F.col(pollutant1+" AQI")).alias(pollutant1),F.avg(F.col(pollutant2+" AQI")).alias(pollutant2))
 df=df.toPandas()
 df.index = pd.to_datetime(df["Date"])
 plt.figure(fig_number)
 df1 = seasonal_decompose(df[pollutant1], model='additive', period=365)
 df1.trend.plot(x='Date', ylim=(0,ylimit),label=pollutant1,color=color1)
 df2 = seasonal_decompose(df[pollutant2], model='additive', period=365)
 df2.trend.plot(x='Date', ylim=(0,ylimit),label=pollutant2,color=color2)
 plt.title("Trend of the Air Quality Index relative to "+ pollutant1+ " and "+pollutant2, fontdict=None, loc='center', pad=None)
 plt.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
 
  
plot_data("NO2","O3",1,40,"blue","orange")
plot_data("SO2","CO",2,8,"green","red")
plt.show()


end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
