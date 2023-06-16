import time
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

start_time=time.time()

df= pd.read_csv('/home/giuseppe/Second_Project/Preprocessed_Dataset.csv')

corr = df.corr();
sb.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,vmax=1,vmin=-1,center=0,cmap='coolwarm',linewidths=0.5)
plt.show()

end_time=time.time()
print("Tempo impiegato:",end_time-start_time,"s")
