import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

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
        

    def show_advanced_eda(data):

        exploration_options = ['-','Count Plot', 'Box Plot']

        if st.checkbox("Show advanced exploration", False):
            st.subheader("Select columns to explore")
            exploration_columns = st.selectbox("Select columns:", (data.columns))

            if st.checkbox("Show count plot", False):
                sns.countplot(x=exploration_columns, data=data).set_title(("{} count plot").format(exploration_columns))
                st.pyplot()

class PlotCharts:
    
    #Plot chats that helps to select features
    def feature_plots(data):
        if st.checkbox("Show Correlation Matrix", False):
            st.subheader("Correlation Matrix")
            corrMatrix = data.corr()
            sns.heatmap(corrMatrix, annot=True)
            st.pyplot()