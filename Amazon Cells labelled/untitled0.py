import pandas as pd
import numpy as np
import sklearn

df=pd.read_csv('amazon_cells_labelled.txt',sep='\t',header=None)

#text cleaning
import re
import nltk
nltk.download('stopwords') #contains list of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

final=[]
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',df[0][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    final.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(final).toarray()
Y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#training with Naive Bayes'
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)

print(accuracy_score(y_test,y_pred))
