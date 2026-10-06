import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


# Load model
model = joblib.load("titanic_model.pkl")


st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger information to predict survival.")


# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Siblings / Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

parch = st.number_input(
    "Parents / Children Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

passenger_class = st.selectbox(
    "Class",
    ["First", "Second", "Third"]
)

who = st.selectbox(
    "Who",
    ["man", "woman", "child"]
)

adult_male = st.selectbox(
    "Adult Male",
    [True, False]
)

alone = st.selectbox(
    "Alone",
    [True, False]
)


# Prediction
if st.button("Predict"):

    data = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "class": passenger_class,
        "who": who,
        "adult_male": adult_male,
        "alone": alone
    }])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("🟢 Survived")
    else:
        st.error("🔴 Not Survived")