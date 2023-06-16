from pyspark.sql import *
from pyspark.sql.functions import substring
from pyspark.sql.functions import countDistinct
from pyspark.sql import functions as F
import time


start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Data Preprocessing") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()


df = spark.read.load("/home/giuseppe/Second_Project/Pollution2000_2016.csv" , format="csv", sep=",", header="true")

df=df.select("State","State Code","County","County Code","City","Site Num","Date Local","NO2 AQI","O3 AQI","SO2 AQI","CO AQI")
df=df.withColumn("Year",substring("Date Local",0,4))
df=df.withColumn("Month",substring("Date Local",6,2))
df=df.withColumn("Day",substring("Date Local",9,2))
df=df.dropna()
df=df.filter(df["Year"]>2009)

df1=df

df1=df1.select("State","County","City","Site Num","Year").groupBy("State","County","City","Site Num").agg(countDistinct("Year").alias("Years"))
df1=df1.filter(df1["Years"]==7)
df1=df1.withColumnRenamed("City","City1")
df1=df1.withColumnRenamed("Years","Years1")

df=df.join(df1, on=["Site Num","State","County"], how="left_semi")


df.coalesce(1).write.option("header","true").format("csv").save("/home/giuseppe/Second_Project/Preprocessed_Dataset")


end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
