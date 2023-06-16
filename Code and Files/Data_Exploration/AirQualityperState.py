from pyspark.sql import *
from pyspark.sql.functions import substring
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql.functions import count
from pyspark.sql.functions import coalesce
from pyspark.sql import functions as F
import time
import pandas as pd
import matplotlib.pyplot as plt



start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Air Quality per State") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()



def plot_data (pollutant,fig_number):
 df_pyspark = spark.read.load("/home/giuseppe/Second_Project/Preprocessed_Dataset.csv" , format="csv", sep=",", header="true")
 df_pyspark=df_pyspark.select((F.to_date("Date Local","yyyy-MM-dd").alias("Date")),"State",pollutant+" AQI")
 df_pyspark1=df_pyspark.groupBy("State","Date").agg(F.avg(F.col(pollutant+" AQI")).alias(pollutant))
 df_pyspark2=df_pyspark1
 df_pyspark3=df_pyspark1
 df_pyspark4=df_pyspark1

 df_pyspark1=df_pyspark1.filter(df_pyspark1[pollutant]<50)
 df_pyspark1=df_pyspark1.groupBy("State").count()
 df_pyspark1=df_pyspark1.select(F.col("State"),F.col("count").alias("Good"))

 df_pyspark2=df_pyspark2.filter((df_pyspark2[pollutant]>=50) & (df_pyspark2[pollutant]<100))
 df_pyspark2=df_pyspark2.groupBy("State").count()
 df_pyspark2=df_pyspark2.select(F.col("State"),F.col("count").alias("Moderate"))

 df_pyspark3=df_pyspark3.filter((df_pyspark3[pollutant]>=100) & (df_pyspark3[pollutant]<150))
 df_pyspark3=df_pyspark3.groupBy("State").count()
 df_pyspark3=df_pyspark3.select(F.col("State"),F.col("count").alias("Unhealthy"))

 df_pyspark4=df_pyspark4.filter(df_pyspark4[pollutant]>=150)
 df_pyspark4=df_pyspark4.groupBy("State").count()
 df_pyspark4=df_pyspark4.select(F.col("State"),F.col("count").alias("Bad"))


 df_final=df_pyspark1.join(df_pyspark2,on="State", how="left_outer")
 df_final=df_final.join(df_pyspark3,on="State", how="left_outer")
 df_final=df_final.join(df_pyspark4, on="State" , how="left_outer")

 df_final=df_final.select("State",coalesce(df_final["Good"],F.lit(0)).alias("Good"),
 coalesce(df_final["Moderate"],F.lit(0)).alias("Moderate"),
 coalesce(df_final["Unhealthy"],F.lit(0)).alias("Unhealthy"),coalesce(df_final["Bad"], F.lit(0)).alias("Bad"))
 df_final=df_final.withColumn("Total Days",df_final["Good"]+df_final["Moderate"]+df_final["Unhealthy"]+df_final["Bad"])
 df_final=df_final.withColumn("Good %",df_final["Good"]*100/df_final["Total Days"])
 df_final=df_final.withColumn("Moderate %",df_final["Moderate"]*100/df_final["Total Days"])
 df_final=df_final.withColumn("Unhealthy %",df_final["Unhealthy"]*100/df_final["Total Days"])
 df_final=df_final.withColumn("Bad %",df_final["Bad"]*100/df_final["Total Days"])

 df_final=df_final.select("State","Good %","Moderate %","Unhealthy %","Bad %")
 df_final=df_final.orderBy("Good %")

 df_final_pandas=df_final.toPandas()
 df_final_pandas.plot.barh(x="State",stacked=True, color=["green","yellow","orange","red"])
 
 plt.title("Average Air Quality Index relative to "+ pollutant+" per State", fontdict=None, loc='center', pad=None)
 plt.figure(fig_number)
 
plot_data("NO2",1)
plot_data("O3",2)
plot_data("SO2",3)
plot_data("CO",4)

plt.show() 

end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
