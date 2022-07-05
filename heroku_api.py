import requests
import json

sample = {'age': 39,
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

# Heroku
r = requests.get("https://ml-fastapi-cj.herokuapp.com/")
print(r.json())
assert r.status_code == 200

r = requests.post("https://ml-fastapi-cj.herokuapp.com/predict",
                  data=json.dumps(sample))

print(r.json()['Income Prediction'])
assert r.status_code == 200

# # Local
# r = requests.get("http://127.0.0.1:8000/")
# print(r.json())
#
# r = requests.post("http://127.0.0.1:8000/predict",
#                   data=json.dumps(sample))
#
# print(r.json())
# print(r.json()['Income Prediction'])

