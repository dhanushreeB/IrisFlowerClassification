import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#load the csv file
df = pd.read_csv("iris.csv")

df.head()
print(df.head())

X=df[["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]]
Y=df["Class"]

#Split the dataset into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=50)

#Scaling
sc =StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#instantiate the model
classifier =RandomForestClassifier()

#fit the model
classifier.fit(X_train,Y_train)

#make pickle file
pickle.dump(classifier,open("model.pkl","wb"))
