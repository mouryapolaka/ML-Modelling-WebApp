import streamlit as st
import pandas as pd
import seaborn as sns

class Exploration:

    #Shows basic exploratory data analysis
    def show_basic_eda(data):
        if st.checkbox("Show basic exploration", False):
            st.subheader("Statistics of each column")
            st.dataframe(data.describe())
            st.subheader("Number of missing values")
            st.dataframe(data.isnull().sum())
            st.subheader("Data type of each column")
            st.dataframe(data.dtypes)

class PlotCharts:
    
    #Plot chats that helps to select features
    def feature_plots(data):
        if st.checkbox("Show Correlation Matrix", False):
            st.subheader("Correlation Matrix")
            corrMatrix = data.corr()
            sns.heatmap(corrMatrix, annot=True)
            st.pyplot()