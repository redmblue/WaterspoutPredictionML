import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

DATA_URL = ('F:\WaterspoutPrediction\Positive.csv')
data234 = pd.read_csv(DATA_URL, 901)
for i in range(0,900):
    currwx = data234['Weather'][i]
    print(currwx)
