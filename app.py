import streamlit as st
import pandas as pd
import numpy as np
import preprocess
import seaborn as sns

def main():
    st.title("DSAI Modelling Platform")
    st.sidebar.image('res/images/dsai_v2.png',width=270)
    st.sidebar.title("Build your Model")
    st.markdown("Build a Classification or Regression Model")
    st.header("1. Import Dataset")
    data_frame = load_data()
    st.header("2. Exploratory Analysis")
    df_stats = show_basic_eda(data_frame)
    st.header("3. Pre Process")
    st.header("4. Feature Selection")
    feature_selection_plots = feature_plots(data_frame)

#Load dataset
def load_data():
    #Prompt user to upload file
    uploaded_file = st.file_uploader("Choose a csv file...", type="csv")

    #If a file is not uploaded, return IRIS dataset
    if uploaded_file is None:
        df = pd.read_csv('res/default_iris.csv')
        if st.checkbox("Show sample dataset (iris)", False):
            st.subheader("Sample Dataset")
            st.dataframe(df)
    else:
        df = pd.read_csv(uploaded_file)
        if st.checkbox("Show uploaded dataset", False):
            st.subheader("Uploaded Dataset")
            st.dataframe(df)

    return df

#Shows basic exploratory data analysis
def show_basic_eda(data):
    if st.checkbox("Show basic exploration", False):
        st.subheader("Statistics of each column")
        st.dataframe(data.describe())
        st.subheader("Number of missing values")
        st.dataframe(data.isnull().sum())
        st.subheader("Data type of each column")
        st.dataframe(data.dtypes)

def feature_plots(data):
    st.subheader("Correlation Matrix")
    corrMatrix = data.corr()
    sns.heatmap(corrMatrix, annot=True)
    st.pyplot()
    
if __name__ == '__main__':
    main()