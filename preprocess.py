import streamlit as st
import pandas as pd
from sklearn import preprocessing
import os.path
import io

class PreProcess:
    def replace_missing_values(data):

        replace_options = ["-","Mean","Median","Mode"]
        st.subheader("Replace Missing Values")
        columns = st.multiselect("Select columns:", (data.columns))
        option = st.selectbox("Replace with:", (replace_options))

        return option, columns

    def feature_encode(data):

        st.subheader("Label Encoder")
        column_selection = st.multiselect("Select columns to encode:", (data.columns))

        return column_selection

    def clean_data(df, missing_value_options, encode_columns):

        data_frame = df
        #Replacing missing values
        option = missing_value_options[0]
        columns = missing_value_options[1]

        if option == 'Mean':
            for cols in columns:
                data_frame[cols].fillna(data_frame[cols].mean(), inplace=True)
        elif option == 'Median':
            for cols in columns:
                data_frame[cols].fillna(data_frame[cols].median(), inplace=True)
        elif option == 'Mode':
            for cols in columns:
                data_frame[cols].fillna(data_frame[cols].mode()[0], inplace=True)

        #Label encoding
        encode_col_selection = encode_columns
        le = preprocessing.LabelEncoder()

        for cols in encode_col_selection:
            data_frame['{}_encoded'.format(cols)] = le.fit_transform(data_frame[cols])

        cleaned_data_path = 'res/cleaned_data.csv'

        if os.path.exists(cleaned_data_path):
            os.remove(cleaned_data_path)
            io.StringIO(df.to_csv(cleaned_data_path,index=False))
        elif os.path.exists(cleaned_data_path) == False:
            io.StringIO(df.to_csv('res/cleaned_data.csv',index=False))

        return data_frame