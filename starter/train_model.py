# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, compute_model_metrics, inference
import joblib

# Add code to load in the data.
data = pd.read_csv('data/clean_census.csv')
print(data.shape)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
print(f"X_train Shape: {X_train.shape}. y_train Shape: {y_train.shape}")

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)
print(f"X_test Shape: {X_test.shape}. y_test Shape: {y_test.shape}")



# Train and save a model.
model = train_model(X_train, y_train)

# Scoring
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print(f"Precision: {precision}, Recall: {recall}, and FBeta: {fbeta}.")

# Saving Model artifacts
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')
joblib.dump(test, 'model/test_set.pkl')