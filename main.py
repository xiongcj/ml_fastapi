import os
from fastapi import FastAPI
from typing import Literal
from pandas import DataFrame
import numpy as np
import uvicorn
from pydantic import BaseModel
from starter.data import process_data
from starter.model import inference
import joblib

# Create app
app = FastAPI()

# POST Input Schema
class ModelInput(BaseModel):
    age: int
    workclass: Literal['State-gov', 'Self-emp-not-inc',
                       'Private', 'Federal-gov',
                       'Local-gov', 'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal['Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc',
                       'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                       'Prof-school', '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int
    marital_status: Literal["Divorced", "Married-spouse-absent",
                            "Never-married", "Married-civ-spouse",
                            "Separated", "Married-AF-spouse", "Widowed"]
    occupation: Literal["Sales", "Exec-managerial", "Prof-specialty",
                        "Tech-support", "Craft-repair", "Other-service",
                        "Farming-fishing", "Transport-moving", "Priv-house-serv",
                        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                        "Protective-serv", "Armed-Forces"]
    relationship: Literal["Not-in-family", "Other-relative", "Unmarried",
                          "Wife", "Own-child", "Husband"]
    race: Literal["White", "Amer-Indian-Eskimo", "Other",
                  "Black", "Asian-Pac-Islander"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
                            'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
                            'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
                            'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                            'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
                            'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
                            'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                            'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                            'Holand-Netherlands']

    # First example in dataset
    class Config:
        schema_extra = {
            "example": {'age': 39,
               'workclass': 'State-gov',
               'fnlgt': 77516,
               'education': 'Bachelors',
               'education_num': 13,
               'marital_status': 'Never-married',
               'occupation': 'Adm-clerical',
               'relationship': 'Not-in-family',
               'race': 'White',
               'sex': 'Male',
               'capital_gain': 2174,
               'capital_loss': 0,
               'hours_per_week': 40,
               'native_country': 'United-States'}
        }


# Load model artifacts
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")


# Root path
@app.get("/")
async def root():
    return {
        "Welcome": "This app predicts whether income exceeds $50K/yr based on census data."}


# Prediction path
@app.post("/predict")
async def predict(input: ModelInput):
    input_dict = input.dict()

    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

    input_df = DataFrame.from_dict(data=input_dict, orient='index').transpose()
    input_df.columns = cols

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    y_pred = inference(model, X)
    y_pred = lb.inverse_transform(y_pred)[0]

    return {"Income Prediction": y_pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
