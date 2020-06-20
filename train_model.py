import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class Train:
    
    #def split_data(features, target):
    def split_data():
        test_size = st.number_input("Test Size", 0.05,1.00, step=0.05, key='test_size')
        random_state = st.number_input("Random State", 1,100, step=1, key='random_state')

        #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state= random_state)
        #return X_train, X_test, y_train, y_test

    def select_algorithm():

        #test_size = split_data()
        mining_problems = ["-","Classification", "Regression"]
        mining_problems_menu = st.selectbox("Select a Mining Problem", (mining_problems))

        if mining_problems_menu == 'Classification':
            classifiers = ["-","Decision Tree","Support Vector Machine (SVM)","Logistic Regression","Random Forest"]
            classifier_menu = st.selectbox("Select a classification algorithm", (classifiers))

            if classifier_menu == 'Decision Tree':
                st.subheader("Decision Tree Parameters")

        elif mining_problems_menu == 'Regression':
            None