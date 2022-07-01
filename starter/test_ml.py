import pandas as pd
import pytest
from starter.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

file_path = 'data/clean_census.csv'
model_file = 'model/model.pkl'

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

@pytest.fixture(name='data')
def data():
    return pd.read_csv(file_path)


def test_load_data(data):
    """
    Check type and shape of data
    """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_model():
    """
    Check model type
    """

    model = joblib.load(model_file)
    assert isinstance(model, RandomForestClassifier)


def test_process_data(data):
    """
    Check train and test dataframes
    """
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == X_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]