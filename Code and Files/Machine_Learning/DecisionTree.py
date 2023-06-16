from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import datetime as dt
import time
import pandas as pd
import matplotlib.pyplot as plt



start_time=time.time()
spark = SparkSession.builder \
.master("local") \
.appName("Decision Tree Regressor") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

df=spark.read.load("/home/giuseppe/Second_Project/Preprocessed_Dataset.csv" , format="csv", sep=",", header="true")
df=df.select("State","County","Site Num","Day","Month","Year",(F.to_date("Date Local","yyyy-MM-dd").alias("Date")),"NO2 AQI","O3 AQI","SO2 AQI","CO AQI")
df=df.orderBy("Date")
df=df.groupBy("Date").agg(avg("NO2 AQI").alias("NO2"),avg("O3 AQI").alias("O3"),avg("SO2 AQI").alias("SO2"),avg("CO AQI").alias("CO"),avg("Day").alias("Day"),avg("Month").alias("Month"),avg("Year").alias("Year"))

def Decision_Tree_Regressor(pollutant,fig_number1,fig_number2,color,dataframe):
 df=dataframe.select("Day","Month","Year","Date",pollutant)
 categoricalColumns = ['Day','Month','Year']
 stages = []
 for categoricalCol in categoricalColumns:
     stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index',handleInvalid='keep')
     encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
     stages += [stringIndexer, encoder]

 assemblerInputs = [c + "classVec" for c in categoricalColumns]
 assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
 stages += [assembler]

 df=df.withColumn(pollutant,df[pollutant].cast("double"))
 
 decision_tree_regressor=DecisionTreeRegressor(featuresCol="features",labelCol=pollutant,maxDepth=20,maxBins=32)

 stages+=[decision_tree_regressor]

 train=df.where(df["Year"]<=2014)
 test=df.where(df["Year"]>2014)
 pipeline=Pipeline(stages=stages)
 PipelineModel=pipeline.fit(train)
 predictions=PipelineModel.transform(test)

 evaluator_RMSE=RegressionEvaluator(labelCol=pollutant,predictionCol="prediction",metricName="rmse")
 evaluator_MAE=RegressionEvaluator(labelCol=pollutant,predictionCol="prediction",metricName="mae")
 rmse=evaluator_RMSE.evaluate(predictions)
 mae=evaluator_MAE.evaluate(predictions)
 
 predictions_Pandas=predictions.toPandas()
 train_Pandas=train.toPandas()
 test_Pandas=test.toPandas()
 
 plt.figure(fig_number1)
 plt.scatter(predictions_Pandas[pollutant],predictions_Pandas["prediction"],color=color)
 plt.xlabel("Real "+pollutant+" values")
 plt.ylabel(pollutant+" prediction")
 plt.title('Predictions for '+pollutant+' with Decision Tree Regressor\nRoot Mean Square Error: '+str(round(rmse,2))+'\nMean Absolute Error: '+str(round(mae,2)))
 
 plt.figure(fig_number2)
 plt.plot(train_Pandas["Date"],train_Pandas[pollutant],color="black",label="Train data")
 plt.plot(test_Pandas["Date"],test_Pandas[pollutant],color="darkgray",label="Test data")
 plt.plot(predictions_Pandas["Date"],predictions_Pandas["prediction"],color=color,label="Predicted data")
 plt.legend(loc='upper left', ncol=1, fancybox=True, shadow=True)
 plt.xlabel("Date")
 plt.ylabel(pollutant+" AQI")
 plt.title('Temporal forecasting for '+ pollutant+ ' with Decision Tree Regressor')


Decision_Tree_Regressor("NO2",1,2,"blue",df)
Decision_Tree_Regressor("O3",3,4,"orange",df)
Decision_Tree_Regressor("SO2",5,6,"green",df)
Decision_Tree_Regressor("CO",7,8,"red",df)

plt.show()

end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
spark.stop();
