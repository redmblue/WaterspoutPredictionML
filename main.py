# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


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
X,y = load_iris(return_X_y=True)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.

#CSV
#Winddir, windspeed, gust factor, vis, conditions, CloudCover, ceiling, temp, dewpt, altimeter
#Winddir(raw), windspeed(raw) , gust factor(null if N/A), Vis(raw).
#Conditions: key: +:0.5 -:0.25 added to thhe base value
#key:
#RA-1
#DZ-2
#SN-3
#SG-4
#IC-5
#PL-6
#GR-6
#GS-7
#UP-8
#FG-9
#BR-10
#HZ-11
#VA-12
#DU-13
#FU-14
#SA-15
#PY-16
#PO-17
#DS-Duststorm
#SS-Sand storm
#FC-Funnel cloud
#Cloud cond: (First layer)
#SKC-1
#None in the file- 0
#NCD-0
#CLR--2
#NSC-3
#FEW-4
#SCT-5
#BKN-6
#OVC-7
#VV-8
#cloud height  - (raw) (first layer)
#temp (raw
#dewpt (raw)
#altimeter(raw, no need for dec)

def getCondValue(cond):
    if cond=='':
        return 0
    extravalue=0
    if cond[0] =='-':
        extravalue=.25
        cond = cond[1:]
    elif cond[0] == '+':
        extravalue=.5
        cond = cond[1:]
    switcher = {
        "RA":1,
        "DZ":2,
        "SN":3,
        "SG":4,
        "IC":5,
        "PL":6,
        "GR":6,
        "GS":7,
        "UP":8,
        "FG":9,
        "BR":10,
        "HZ":11,
        "VA":12,
        "DU":13,
        "FU":14,
        "SA":15,
        "PY":16,
        "PO":17
    }
    return switcher.get(cond, 0)+extravalue
#RA - 1
#-RA - 1.25
#+RA - 1.50
def getCloudCover(cc):
    if cc=='':
        return 0
    switcher={
        'SKC':1,
        'NCD':0,
        'CLR':2,
#       'NSC':3,
        'FEW':4,
        'SCT':5,
        'BKN':6,
        'OVC':7,
        'VV':8

    }
    key=cc
    return(switcher.get(key,"default"))
#print("Hello from a function")
windira=[]
windspda=[]
gustfactora=[]
visa=[]
conda=[]
cldcvra=[]
ceilinga=[]
tempa=[]
dewa=[]
alta=[]
conde = []
# wxe.append(windir+ " " +windspd+ " " +gustfactor+ " " +vis+ " " +cond+ " " +cldcvr+ " " +ceiling+ " " +temp+ " " +dew+ " " +alt)
with open('F:\\WaterspoutPrediction\\RawWaterspoutData.csv', 'r') as csvfile:
    datareader = csv.reader(csvfile)
    sd= 0
    for row in datareader:
        try:
           sd+=1
           if sd==1:
             continue
           print(row[2])
           arr = row[4].split(" ")
           print(arr)
           windir=arr[3][:3]
           if(windir=='///'):
               windir=0
           windspd=arr[3][3:5]
           if(arr[3][5]=='G'):
               gustfactor=arr[3][6:-2]
           else:
               gustfactor=0
           if(len(arr[4]))==3 or (len(arr[4]))==4:
                vis=arr[4][:-2]
           else:
               if(arr[4][:-2])==2:
                   vis=0.5
               elif(arr[4][:-2])==4:
                   vis=0.25
               elif(arr[4][:-2])==8:
                   vis=0.125
               else:
                   vis=0
           if "RA" in arr[5] or "DZ" in arr[5] or "SN" in arr[5] or "SG" in arr[5] or "IC" in arr[5] or "PL" in arr[5] or "GR" in arr[5]or "GS" in arr[5] or "UP" in arr[5] or "FG" in arr[5] or "BR" in arr[5] or "HZ" in arr[5]  or"VA" in arr[5] or "DU" in arr[5] or "FU" in arr[5] or "SA" in arr[5] or "PY" in arr[5] or "PO" in arr[5]:
               #clcd=arr[4][:2]
               cond=getCondValue(arr[5])
               #clht=arr[4][2:]
               i=6
               while "RA" in arr[i] or "DZ" in arr[i] or "SN" in arr[i] or "SG" in arr[i] or "IC" in arr[i] or "PL" in arr[i] or "GR" in arr[i] or "GS" in arr[i] or "UP" in arr[i] or "FG" in arr[i] or "BR" in arr[i] or "HZ" in arr[i] or "VA" in arr[i] or "DU" in arr[i] or "FU" in arr[i] or "SA" in arr[i] or "PY" in arr[i] or "PO" in arr[i]:
                   i+=1

               if ('SKC' in arr[i]) or ('NCD' in arr[i]) or ('CLR' in arr[i]) or ('NSC' in arr[i]) or ('FEW' in arr[i])  or ('SCT' in arr[i])  or ('BKN' in arr[i])  or ('OVC' in arr[i])  or ('VV' in arr[i]):
                   if 'VV' in arr[i]:
                       cldcvr = 8
                       ceiling = arr[i][-3:]
                       i += 1
                   else:
                       cldcvr = getCloudCover(arr[i][:3])
                       if(cldcvr==2):
                           ceiling=0
                       else:
                         ceiling = arr[i][-3:]
                       i += 1
               else:
                   cldcvr = 0
                   ceiling = 0
               while 'SKC' in arr[i] or 'NCD' in arr[i] or 'CLR' in arr[i] or 'NSC' in arr[i] or 'FEW' in arr[
                   i] or 'SCT' in arr[i] or 'BKN' in arr[i] or 'OVC' in arr[i] or 'VV' in arr[i]:
                   i += 1
               aasr=arr[i].split("/")
               if('M' in aasr[0]):
                   aasr[0] = '-'+aasr[0][-2:]
               if('M' in aasr[1]):
                   aasr[1] = '-'+aasr[1][-2:]
               temp=aasr[0]
               dew=aasr[1]
               i += 1
               alt = arr[i][1:]
               print(windir,windspd,gustfactor,vis,cond,cldcvr,ceiling,temp,dew,alt)
               if(alt=='MK'):
                   continue
               windira.append(windir)
               windspda.append(windspd)
               gustfactora.append(gustfactor)
               visa.append(vis)
               conda.append(cond)
               cldcvra.append(cldcvr)
               ceilinga.append(ceiling)
               tempa.append(temp)
               dewa.append(dew)
               alta.append(alt)
               conde.append(row[5])
               #conde.append(row[5])
           else:
               cond=0
               i=6
               if ('SKC' in arr[i]) or ('NCD' in arr[i]) or ('CLR' in arr[i]) or ('NSC' in arr[i]) or ('FEW' in arr[i])  or ('SCT' in arr[i])  or ('BKN' in arr[i])  or ('OVC' in arr[i])  or ('VV' in arr[i]):
                   if 'VV' in arr[i]:
                       cldcvr = 8
                       ceiling = arr[i][-3:]
                       i += 1
                   else:
                       cldcvr = getCloudCover(arr[i][:3])
                       if(cldcvr==2):
                           ceiling=0
                       else:
                         ceiling = arr[i][-3:]
                       i += 1
               else:
                   cldcvr = 0
                   ceiling = 0
               while 'SKC' in arr[i] or 'NCD' in arr[i] or 'CLR' in arr[i] or 'NSC' in arr[i] or 'FEW' in arr[
                   i] or 'SCT' in arr[i] or 'BKN' in arr[i] or 'OVC' in arr[i] or 'VV' in arr[i]:
                   i += 1
               aasr=arr[i].split("/")
               if('M' in aasr[0]):
                   aasr[0] = '-'+aasr[0][-2:]
               if('M' in aasr[1]):
                   aasr[1] = '-'+aasr[1][-2:]
               temp=aasr[0]
               dew=aasr[1]
               i += 1
               alt = arr[i][1:]
               print(windir,windspd,gustfactor,vis,cond,cldcvr,ceiling,temp,dew,alt)
               if(alt=='MK'):
                   continue
               #wxe.append(windir+ " " +windspd+ " " +gustfactor+ " " +vis+ " " +cond+ " " +cldcvr+ " " +ceiling+ " " +temp+ " " +dew+ " " +alt)
               windira.append(windir)
               windspda.append(windspd)
               gustfactora.append(gustfactor)
               visa.append(vis)
               conda.append(cond)
               cldcvra.append(cldcvr)
               ceilinga.append(ceiling)
               tempa.append(temp)
               dewa.append(dew)
               alta.append(alt)
               conde.append(row[5])
        except IndexError:
            continue
          #print("placeholder")
          #clouds - if no clouds, then temp
       #print decoded/list form
print(len(windira))
print(len(windspda))
print(len(gustfactora))
print(len(visa))
print(len(conda))
print(len(cldcvra))
print(len(ceilinga))
print(len(tempa))
print(len(dewa))
print(len(alta))
print(len(conde))
csv_make = {'Winddir':windira,'Windspeed': windspda,'Gust_Factor':gustfactora,'visability':visa ,'Condition':conda, 'cloudCover': cldcvra, 'ceiling': ceilinga, 'temp': tempa, 'dew':dewa, 'alt':alta, 'Conditions':conde}
dfev = pd.DataFrame(csv_make)
dfev.to_csv('rawdatawaterspout.csv')




#
#df = pd.read_csv('F:\\WaterspoutPrediction\\RawWaterspoutData.csv')
#y=df['Condition']
#x=df.drop(['Condition'],axis=1)
#
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
#clf=MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=500,alpha=0.0001,solver='sgd',
#                  verbose=19,random_state=21,tol=0.000000001)
#clf.fit(x_train,y_train)
#y_pred = clf.predict(x_test)
#accuracy_score(y_test,y_pred)
#cm=confusion_matrix(y_test,y_pred)





#sns.heatmap(cm,center=True)
#plt.show()
#cm
#DATA_URL = ('F:\\WaterspoutPrediction\\Positive.csv')
#data234 = pd.read_csv(DATA_URL, usecols=[1])
##for i in range(1,100):
#    #currwx = data234['Weather']['KCSM']
#    #print(currwx)
#def crawl(x):
#   print(x)
#data234.apply(crawl)
## Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#   #print_hi('PyCharm')
#   ##clf = SVC()
#   ##clf.set_params(kernel='linear').fit(X,y)
#
#   #regressor = linear_model.LinearRegression()
#   #regressor.fit(X,y)
#
#   #plott.scatter(X,y, color='blue')
#
#   # Load the diabetes dataset
#   diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
#
#   # Use only one feature
#   diabetes_X = diabetes_X[:, np.newaxis, 2]
#
#   # Split the data into training/testing sets
#   diabetes_X_train = diabetes_X[:-20]
#   diabetes_X_test = diabetes_X[-20:]
#
#   # Split the targets into training/testing sets
#   diabetes_y_train = diabetes_y[:-20]
#   diabetes_y_test = diabetes_y[-20:]
#
#   # Create linear regression object
#   #regr = linear_model.LinearRegression()
#   regr = MLPRegressor(random_state=1, max_iter=500).fit(diabetes_X_train, diabetes_y_train)
#   # Train the model using the training sets
#   #regr.fit(diabetes_X_train, diabetes_y_train)
#
#   # Make predictions using the testing set
#   diabetes_y_pred = regr.predict(diabetes_X_test)
#
#   # The coefficients
#   print('Coefficients: \n', regr.coef_)
#   # The mean squared error
#   print('Mean squared error: %.2f'
#         % mean_squared_error(diabetes_y_test, diabetes_y_pred))
#   # The coefficient of determination: 1 is perfect prediction
#   print('Coefficient of determination: %.2f'
#         % r2_score(diabetes_y_test, diabetes_y_pred))
#
#   # Plot outputs
#   plott.scatter(diabetes_X_test, diabetes_y_test, color='black')
#   plott.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
#
#   plott.xticks(())
#   plott.yticks(())
#
#   plott.show()
#
## See PyCharm help at https://www.jetbrains.com/help/pycharm/
#