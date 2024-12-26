import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor


def model_training_and_evaluation():
    st.title("Model Training and Evaluation")
    st.subheader("Train multiple machine learning models on your dataset")

    if "data" in st.session_state:
        df = st.session_state["data"]

        # Data Overview Section (one column)
        st.title("Data Overview")
        st.write("Dataset Preview:")
        st.dataframe(df)

        # Feature(X) and Target Selection(Y)
        features = st.multiselect("Select Feature Columns", options=df.columns)
        target = st.selectbox("Select Target Column", options=df.columns)

        # Train-Test Split Percentage
        train_size = st.slider("Select % to train model",
                               min_value=1, max_value=99, value=80) / 100

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine (SVM)": SVC(),
            "XGBoost Classifier": XGBClassifier(),
            "XGBoost Regressor": XGBRegressor()

        }
        selected_models = st.multiselect(
            "Select Models to Train", options=list(models.keys()))

        # Train and Evaluate Models when button clicked
        if st.button("Train Models"):
            if not features or not target:
                st.error("Please select both feature columns and a target column.")
            else:
                try:
                    # Prepare Data
                    X = df[features]
                    y = df[target]

                    # Check for categorical features
                    if X.select_dtypes(include=["object"]).shape[1] > 0:
                        st.info(
                            "Detected categorical variables in features. Encoding them automatically...")
                        X = pd.get_dummies(X, drop_first=True)

                    if y.dtype == "object":
                        st.info(
                            "Detected categorical target. Encoding it automatically...")
                        y = pd.factorize(y)[0]

                    # Train-Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, train_size=train_size, random_state=42)

                    # Scaling Data
                    scaler = StandardScaler()  # std, Mean
                    X_train = pd.DataFrame(scaler.fit_transform(
                        X_train), columns=X_train.columns)
                    X_test = pd.DataFrame(scaler.transform(
                        X_test), columns=X_test.columns)

                    # Train and Evaluate Models
                    st.subheader("Model Results")
                    for model_name in selected_models:
                        st.write(f"### {model_name}")
                        model = models[model_name]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # Evaluation Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                        st.write(f"Accuracy: {accuracy:.2%}")
                        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                        st.text("Confusion Matrix:")
                        st.write(confusion_matrix(y_test, y_pred))

                    st.success("Model Training Completed!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("No data available. Please upload or load a dataset.")
