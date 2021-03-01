import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
st.title("WaterSpout Prediction")
DATE_COLUMN = 'Airport'
DATA_URL = ('F:\WaterspoutPrediction\Positive.csv')
DATE_COLUMN1 = 'Airport'
DATA_URL1 = ('F:\\WaterspoutPrediction\\Negative.csv')
data234 = pd.read_csv(DATA_URL, 901)
#for i in range(0,900):
 #   currwx = data234['Weather'][i]
def load_data_pos(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
def load_data_neg(nrows):
    data = pd.read_csv(DATA_URL1, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
def checkbox():
    checkbox1 = st.checkbox('Show pos raw data')
    st.checkbox('Test')
    checkbox2 = st.checkbox('Show neg raw data')
    if checkbox1:
        st.subheader('Raw data')
        st.write(data)
    if checkbox2:
        st.subheader('Raw data')
        st.write(data1)
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data_pos(901)
data1 = load_data_neg(345)
data_load_state.text("Data has been loaded")
#readme_text = st.markdown("Data:")
#checkbox1 = st.write('Test')
#checkbox2=st.write('Test')
#st.sidebar.title("What to do")
#st.write("Data:")
st.sidebar.write("Raw Data:")
    #checkbox()
if(st.sidebar.checkbox('Show pos raw data')):
    st.subheader('Pos Raw data')
    st.write(data)
if(st.sidebar.checkbox('Show neg raw data')):
    st.subheader('Neg Raw data')
    st.write(data1)
    #st.sidebar.success('To look at the algorithms select "algorithms".')
