import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
import pandas as pd

#import classifiers from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#import regressors from sklearn


class Train:
    
    def feature_selection(data):

        feature_cols = st.multiselect("Select features", (data.columns))
        target_col = st.selectbox("Select target:", (data.columns))

        return data[feature_cols], data[target_col]

    #def split_data(features, target):
    def split_data(x):

        X= x[0]
        y= x[1]
        class_names = y.unique()

        test_size = st.number_input("Test Size", 0.05,1.00, step=0.05, key='test_size')
        random_state = st.number_input("Random State", 1,100, step=1, key='random_state')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state= random_state)
        
        return X_train, X_test, y_train, y_test, class_names

    def train_data(split_df):

        X_train = split_df[0]
        X_test = split_df[1]
        y_train = split_df[2]
        y_test = split_df[3]
        target_names = split_df[4]

        mining_problems = ["-","Classification", "Regression"]
        mining_problems_menu = st.selectbox("Select a Mining Problem", (mining_problems))

        if mining_problems_menu == 'Classification':
            st.write("List of classes")
            st.dataframe(target_names)
            classifiers = ["-","Decision Tree","Support Vector Machine (SVM)","Logistic Regression","Random Forest"]
            classifier_menu = st.selectbox("Select a classification algorithm", (classifiers))
            
            #Classification for decision tree
            if classifier_menu == 'Decision Tree':
                st.subheader("Decision Tree Parameters")
                max_depth = st.slider("Max depth of the tree",1,100, key="max_depth")
                min_samples_split = st.slider("Minimum samples split",1,100, key="min_samples_split")

                if st.button("classify", key='classify'):
                    st.subheader("Decision tree results")
                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    
                    if len(target_names) > 2:
                        st.write("Precision: ", precision_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                    else:
                        st.write("Precision: ", precision_score(y_test,y_pred, labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, labels=target_names).round(2))
                    
                    '''
                    class_report = classification_report(y_pred,y_test,output_dict=True)
                    class_report_df = pd.DataFrame(class_report)

                    if st.checkbox("Show classification report", False):
                        st.dataframe(class_report_df)
                    '''
            
            #Classification for SVM
            if classifier_menu == 'Support Vector Machine (SVM)':
                st.subheader("SVM Parameters")
                c_value = st.slider("C value",1,100, key="c_value")
                kernel_value = st.selectbox("Select kernel value", ['linear','poly','rbf','sigmoid','precomputed'])

                if st.button("classify", key='classify'):
                    st.subheader("SVM results")
                    model = SVC(C=c_value, kernel=kernel_value)
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    
                    if len(target_names) > 2:
                        st.write("Precision: ", precision_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                    else:
                        st.write("Precision: ", precision_score(y_test,y_pred, labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, labels=target_names).round(2))
                    
                    '''
                    class_report = classification_report(y_pred,y_test,output_dict=True)
                    class_report_df = pd.DataFrame(class_report)

                    if st.checkbox("Show classification report", False):
                        st.dataframe(class_report_df)
                    '''

            #Classification for logistic regression
            if classifier_menu == 'Logistic Regression':
                st.subheader("Logistic Regression Parameters")
                c_value = st.slider("C value",1,100, key="c_value")
                solver_value = st.selectbox("Select solver", ['newton-cg','lbfgs','liblinear','sag','saga'])
                max_iter_value = st.slider("Maximum Iterations",1,500, key="max_iter_value")
                multi_class_value = st.selectbox("Select multi_class", ['auto','ovr','multinomial'])

                if st.button("classify", key='classify'):
                    st.subheader("Logistic Regression results")
                    model = LogisticRegression(C=c_value, solver=solver_value, max_iter=max_iter_value, multi_class=multi_class_value)
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    
                    if len(target_names) > 2:
                        st.write("Precision: ", precision_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                    else:
                        st.write("Precision: ", precision_score(y_test,y_pred, labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, labels=target_names).round(2))
                    
                    '''
                    class_report = classification_report(y_pred,y_test,output_dict=True)
                    class_report_df = pd.DataFrame(class_report)

                    if st.checkbox("Show classification report", False):
                        st.dataframe(class_report_df)
                    '''

            #Classification for Random Forest     
            if classifier_menu == 'Random Forest':
                st.subheader("Random Forest Parameters")
                n_estimators = st.slider("N-estimators (Number of trees)",1,500, key="n_estimators")
                max_depth = st.slider("Max depth of the tree",1,100, key="max_depth")
                min_samples_split = st.slider("Minimum samples split",1,100, key="min_samples_split")

                if st.button("classify", key='classify'):
                    st.subheader("Random Forest results")
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    
                    if len(target_names) > 2:
                        st.write("Precision: ", precision_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, average = 'weighted', labels=target_names).round(2))
                    else:
                        st.write("Precision: ", precision_score(y_test,y_pred, labels=target_names).round(2))
                        st.write("Recall: ", recall_score(y_test,y_pred, labels=target_names).round(2))

        elif mining_problems_menu == 'Regression':
            regressors = ["-","Decision Tree","Support Vector Machine (SVM)","Logistic Regression","Random Forest"]
            regressor_menu = st.selectbox("Select a regression algorithm", (regressors))