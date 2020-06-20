import streamlit as st
import pandas as pd

class MissingValues:

    def replace_missing_values(data):
        st.subheader("Replace Missing Values")
        columns = st.multiselect("Select columns", (data.columns))
        options = st.selectbox("Replace with:", ("Choose an option","Mean","Median","Mode"))

        if options == 'Mean':
            for cols in columns:
                data.fillna(data[cols].mean())
        elif options == 'Median':
            pass
        elif options == 'Mode':
            pass
        
        return data

class LabelEncode:

    def feature_encode(dataset):
        st.subheader("Label Encode")
        encode_columns = st.multiselect("Select columns to encode:", (dataset.columns))