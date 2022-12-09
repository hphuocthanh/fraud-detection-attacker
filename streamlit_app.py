import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

from io import StringIO

st.set_page_config(
    page_title="Fraud Detection Prediction", page_icon="📶", layout="wide"
)

st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True,
)

h = st.markdown(
    """
<style>
div.fullScreenFrame > div {
    display: flex;
    justify-content: center;
}
</style>""",
    unsafe_allow_html=True,
)

# Title
original_title = '<p style="text-align: center; color:#3498DB; text-shadow: 2px 2px 4px #000000; font-size: 60px;">Dự Đoán Khả Năng Lừa Đảo Tín Dụng</p>'
st.markdown(original_title, unsafe_allow_html=True)

# Reads in saved Regression model
load_clf = pickle.load(open("model.pkl", "rb"))

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

    # Apply model to make predictions
    prediction = load_clf.predict(dataframe)
