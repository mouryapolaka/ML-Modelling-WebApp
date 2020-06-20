import streamlit as st
import pandas as pd
import numpy as np
import preprocess as PP
import data_exploration as DE

def main():
    st.title("DSAI Modelling Platform")
    st.markdown("Build a Classification or Regression Model")
    st.sidebar.image('res/images/dsai_v2.png',width=300)
    st.sidebar.markdown("Build a classification or regression model using DSAI platform.")

    #Import dataset
    st.header("1. Import Dataset")
    data_frame = load_data()
    temp_df = data_frame.copy()

    #Exploratory analysis section
    st.header("2. Exploratory Analysis")
    df_stats = DE.Exploration.show_basic_eda(data_frame)

    #Data cleaning section
    st.header("3. Pre Process")
    clean_missing_values = PP.PreProcess.replace_missing_values(data_frame)
    feature_encoding = PP.PreProcess.feature_encode(data_frame)

    if st.button("Clean Data", key = 'clean_data'):
        clean_btn = PP.PreProcess.clean_data(temp_df,clean_missing_values,feature_encoding)
        st.subheader("Pre Processed Data Frame")
        st.dataframe(clean_btn)
    
    #Feature selection section
    st.header("4. Feature Selection")
    feature_selection_plots = DE.PlotCharts.feature_plots(data_frame)

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

if __name__ == '__main__':
    main()