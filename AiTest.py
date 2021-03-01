import numpy as np
from sklearn.datasets import load_iris
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plott
import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv
df = pd.read_csv('rawdatawaterspout.csv')
y=df['Conditions']
x=df.drop(['Conditions'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
clf=MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=500,alpha=0.0001,solver='adam',
                  verbose=19,random_state=21,tol=0.000000001)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)