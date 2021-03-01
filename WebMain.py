import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import altair as alt
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
        "PO":17,
        "None":0
    }
    return switcher.get(cond, 0)+extravalue
def getCloudCover(cc):
    if cc=='':
        return 0
    switcher={
        'SKC':1,
        'NCD':0,
        'None':0,
        'CLR':2,
        'NSC':3,
        'FEW':4,
        'SCT':5,
        'BKN':6,
        'OVC':7,
        'VV':8

    }
    key=cc
    return(switcher.get(key,"default"))

#from pandas.tools.plotting import scatter_matrix
st.title("WaterSpout Prediction")
df= pd.read_csv('rawdatawaterspout.csv')

#first make some fake data with same layout as yours
data = pd.DataFrame(np.random.randn(100, 11), columns=['Winddir','Windspeed','Gust_Factor','visability','Condition','cloudCover','ceiling','temp','dew','alt','Conditions'])

#,Winddir,Windspeed,Gust_Factor,visability,Condition,cloudCover,ceiling,temp,dew,alt,Conditions

df = pd.read_csv('rawdatawaterspout.csv')
y=df['Conditions']
x=df.drop(['Conditions','index'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
clf=MLPClassifier(hidden_layer_sizes=(100,100,100),activation='relu',max_iter=500,alpha=0.0001,solver='lbfgs',
                  verbose=19,random_state=21,tol=0.000000001)
#clf.fit(x_train,y_train)
#y_pred = clf.predict(x_test)
##print(x_test)
#cm=confusion_matrix(y_test,y_pred)
Wind_dir = st.slider("Wind direction",0,359,1)
Wind_speed = st.slider("Wind speed",0,99,1)
Gust_factor = st.slider("Gust factor",0,99,1)
Visibility = st.slider("Visibility",0,10,1)
skycond = st.selectbox(
     'Sky Conditions',
    ['None','RA', 'DZ', 'SN', 'SG','IC','PL','GR','GS','UP','FG','BR','HZ','VA','DU','FU','SA','PY','PO'])

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
#'SKC-1'
#'None in the file- 0'
#'NCD-0'
#'CLR--2'
#'NSC-3'
#'FEW-4'
#'SCT-5'
#'BKN-6'
#'OVC-7'
#'VV-8'
#cloud height  - (raw) (first layer)
#temp (raw
#dewpt (raw)
#altimeter(raw, no need for dec)
cloudcover = st.selectbox(
     'Cloud Cover',
    ['None','SKC','NCD','CLR','NSC','FEW','SCT','BKN','OVC','VV'])

ceiling = st.text_input("Ceiling(in feet)(0=unlimted)")
temp = st.slider("Temperature",-30,100,1)
dew = st.slider("Dewpoint",-30,100,1)
altim=st.text_input("Altimeter(no decimal)")
print(Wind_dir)
print(Wind_speed)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

st.write(accuracy_score(y_test, y_pred))
try:
    predict = clf.predict(np.array(
        [float(Wind_dir), float(Wind_speed), float(Gust_factor), float(Visibility), float(getCondValue(skycond)),
         float(getCloudCover(cloudcover)), float(ceiling), float(temp), float(dew), float(altim)]).reshape(1, -1))
    st.write(predict)
    #st.write(clf.predict(np.array([340,15,20,10,0,0,0,-1,-8,3014]).reshape(1,-1)))
except ValueError:
    st.write("Please Enter all Fields")
#now plot using pandas
#pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
#c = alt.Chart(df).mark_circle().encode()
#st.altair_chart(c, use_container_width=True)