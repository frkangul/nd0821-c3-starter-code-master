from starter.tests.data import process_data
from starter.tests.model import compute_model_metrics, inference, train_model
import pandas as pd

file_path = "/starter/data/census_cleaned.csv"
data = pd.read_csv(file_path)


def test_data_shape(data):
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_process_data(data):
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
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert y.shape[0] > 0


def test_inference(data):
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
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    clf = train_model(X, y)
    preds = inference(clf, X)

    assert preds.shape[0] > 0
    return y, preds


def test_compute_metric(data):
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
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    clf = train_model(X, y)
    preds = inference(clf, X)

    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

