import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv('data.csv')
df.info()
df.isnull().sum()
df.drop(df.columns[10:],inplace=True,axis=1)
df.dropna(inplace=True)
df.isnull().sum()

X=df.iloc[:,1:9].values
Y=df.iloc[:,9].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelen=LabelEncoder()
onehoten=OneHotEncoder(categorical_features=[0,2,3])

X[:,0]=labelen.fit_transform(X[:,0])
X[:,2]=labelen.fit_transform(X[:,2])
X[:,3]=labelen.fit_transform(X[:,3])
X=onehoten.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

