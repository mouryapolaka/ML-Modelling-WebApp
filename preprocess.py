import streamlit as st
import pandas as pd
from sklearn import preprocessing

class PreProcess:

    def replace_missing_values(data):

        replace_options = ["-","Mean","Median","Mode"]
        st.subheader("Replace Missing Values")
        columns = st.multiselect("Select columns", (data.columns))
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
            pass
        elif option == 'Mode':
            pass

        #Label encoding
        encode_col_selection = encode_columns
        le = preprocessing.LabelEncoder()

        for cols in encode_col_selection:
            data_frame['{}_encoded'.format(cols)] = le.fit_transform(data_frame[cols])

        return data_frame