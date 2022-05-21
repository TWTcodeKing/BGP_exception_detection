from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

df = pd.read_csv('./data/nimda.csv')

df_value = df.values
X,y = df_value[:,:-1],df_value[:,-1]
for i in range(X.shape[1]):
    b,a = signal.butter(8,0.2,'lowpass')
    X[:,i] = signal.filtfilt(b,a,X[:,i])
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.6,test_size=0.4,random_state=42)


Model = GaussianNB()
Model.fit(X_train,y_train)


y_predprob = Model.predict_proba(X_test)[:,1]

y_predprob = y_predprob-0.5
for index,value in enumerate(y_predprob):
    if value<0:
        y_predprob[index] = 0
    else:
        y_predprob[index] = 1

print("acc: %f" % metrics.accuracy_score(y_test,y_predprob))
print("recall:%f" % metrics.recall_score(y_test,y_predprob))
print("precision: %f" % metrics.precision_score(y_test,y_predprob))
print("F1:%f" % metrics.f1_score(y_test,y_predprob))