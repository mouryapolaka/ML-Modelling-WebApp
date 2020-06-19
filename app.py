import streamlit as st
import pandas as pd
import numpy as np
import preprocess
import data_exploration as DE

def main():
    st.title("DSAI Modelling Platform")
    st.sidebar.image('res/images/dsai_v2.png',width=270)
    st.sidebar.title("Build your Model")
    st.markdown("Build a Classification or Regression Model")

    st.header("1. Import Dataset")
    data_frame = load_data()

    st.header("2. Exploratory Analysis")
    df_stats = DE.Exploration.show_basic_eda(data_frame)

    st.header("3. Pre Process")
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