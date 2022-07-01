from fastapi.testclient import TestClient
from main import app
import json

# Instantiate the testing client with our app.
client = TestClient(app)

def test_root():
    """
    Test root page for welcome message.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "Welcome": "This app predicts whether income exceeds $50K/yr based on census data."}


def test_post_low_income():
    """
    Test an example when income is less than 50K
    """

    sample_1 = {'age': 39,
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
                 'native_country': 'United-States',
                 'salary': '<=50K'}
    sample_1.pop('salary')

    r = client.post("/predict", json=sample_1)

    assert r.status_code == 200
    assert r.json() == {"Income Prediction": "<=50K"}


def test_post_high_income():
    """
    Test an example when income is higher than 50K
    """

    sample_2 = {'age': 52,
                'workclass': 'Self-emp-inc',
                'fnlgt': 287927,
                'education': 'HS-grad',
                'education_num': 9,
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Wife',
                'race': 'White',
                'sex': 'Female',
                'capital_gain': 15024,
                'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': 'United-States',
                'salary': '>50K'}
    sample_2.pop('salary')

    r = client.post("/predict", json=sample_2)

    assert r.status_code == 200
    assert r.json() == {"Income Prediction": ">50K"}
