# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from starter.ml.model import inference
from starter.ml.data import process_data
from typing import Literal
import numpy as np
import pandas as pd
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class req_body(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']


app = FastAPI()


@app.get("/")
async def welcome_msg():
    return {"msg": "Welcome salary prediction application"}


@app.post("/")
async def model_inference(test_data: req_body):
    model = load("model/GradientBoostingClassifier.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    input_array = np.array([[
                     test_data.age,
                     test_data.workclass,
                     test_data.education,
                     test_data.marital_status,
                     test_data.occupation,
                     test_data.relationship,
                     test_data.race,
                     test_data.sex,
                     test_data.hours_per_week,
                     test_data.native_country
                     ]])

    df_temp = pd.DataFrame(data=input_array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    X, _, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    print(X)
    data_deneme = pd.DataFrame(X)
    print(data_deneme.sum())
    preds = inference(model, X)
    print(df_temp)
    print(preds)
    y = lb.inverse_transform(preds)[0]
    return {"prediction": y}