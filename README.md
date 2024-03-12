# KNN-using-Sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
df=pd.read_csv("/content/sample_data/ClassifiedData (1).csv",index_col=0)
scaler  = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
x_train,x_test,y_train,y_test=train_test_split(scaled_features,
df['TARGET CLASS'],test_size=0.30)
#intially with k=1
knn1= KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train,y_train)
pred1=knn1.predict(x_test)
print("For K=1 results are:")
print(classification_report(y_test,pred1))
#NOW WITH K=23
knn23=KNeighborsClassifier(n_neighbors=23)
knn23.fit(x_train,y_train)
pred23 = knn23.predict(x_test)
print("For k=23  result are:")
print(classification_report(y_test,pred23))
#NOW WITH K=23
knn25=KNeighborsClassifier(n_neighbors=10)
knn25.fit(x_train,y_train)
pred25=knn25.predict(x_test)
print("For K=25 results are:")
print(classification_report(y_test,pred25))






