import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

st.set_page_config(page_title="ML Prediction App", page_icon="🌸")

st.title("🌸 Machine Learning Prediction App")
st.write("Experiment 6: Train, Evaluate, and Deploy a Machine Learning Model")

st.header("Enter Flower Measurements")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

if st.button("Predict Flower"):
    data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(data)[0]
    iris = load_iris()
    flower_name = iris.target_names[prediction]

    st.success(f"Predicted Flower: {flower_name.title()}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        st.subheader("Prediction Confidence")
        for name, probability in zip(iris.target_names, probabilities):
            st.write(f"{name.title()}: {probability * 100:.2f}%")
