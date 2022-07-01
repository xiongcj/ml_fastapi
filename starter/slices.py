import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import compute_model_metrics
import joblib

# Categorical features
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

model_path = 'model'
file_path ='data/clean_census.csv'

def test_slice():
    """
    Outputs the performance on slices of categorical features
    """

    data = pd.read_csv(file_path)
    _, test = train_test_split(data, test_size=0.20)

    rf = joblib.load(os.path.join(model_path, 'model.pkl'))
    encoder = joblib.load(os.path.join(model_path, 'encoder.pkl'))
    lb = joblib.load(os.path.join(model_path, 'lb.pkl'))

    slices_output = []

    for feature in cat_features:
        for cls in test[feature].unique():
            df_temp = test[test[feature] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,cat_features,label="salary",
                training=False,encoder=encoder,lb=lb)

            y_pred = rf.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f"{feature} ({cls}) :--: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slices_output.append(row)

            with open('slices_output/slice_output.txt', 'w') as file:
                for row in slices_output:
                    file.write(row + '\n')


if __name__ == '__main__':
    test_slice()
