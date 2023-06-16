from pyspark.sql import *
from pyspark.sql.functions import substring
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql import functions as F
import time
import pandas as pd
import matplotlib.pyplot as plt



start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Average AQIs") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

df = spark.read.load("/home/giuseppe/Second_Project/Preprocessed_Dataset.csv" , format="csv", sep=",", header="true")

df=df.select((F.to_date("Date Local","yyyy-MM-dd").alias("Date")),"NO2 AQI","O3 AQI","SO2 AQI","CO AQI")
df=df.orderBy("Date")
df=df.groupBy("Date").agg(F.avg(F.col("NO2 AQI")).alias("NO2"),F.avg(F.col("O3 AQI")).alias("O3"),F.avg(F.col("SO2 AQI")).alias("SO2"),F.avg(F.col("CO AQI")).alias("CO"))
df=df.toPandas()


df.plot(x='Date')
plt.title('US average AQIs per pollutant', fontdict=None, loc='center', pad=None)
plt.xlabel('Date')
plt.ylabel('Average AQI')
plt.legend(loc='upper left', ncol=1, fancybox=True, shadow=True)
plt.show()

end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
