# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import joblib


# Add code to load in the data.
data = pd.read_csv("../data/census_cleaned.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
# drop unnecessary columns
data.drop(columns=["fnlgt", "education-num", "capital-gain", "capital-loss"], inplace=True)
train, test = train_test_split(data, test_size=0.20)

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

# Train and save a model.
clf = train_model(X_train, y_train)
preds = inference(clf, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, preds)
print(f"precision = {precision}, recall = {recall}, fbeta = {fbeta}")
# Save model
joblib.dump(clf, "../model/GradientBoostingClassifier.joblib")

# Train model on slices of data
sliced_values = []

for cat in cat_features:
    for cls in test[cat].unique():
        df_temp = test[test[cat] == cls]

        X_test, y_test, _, _ = process_data(
            df_temp,
            categorical_features=cat_features,
            label="salary", encoder=encoder, lb=lb, training=False)

        y_preds = clf.predict(X_test)

        prc, rcl, fb = compute_model_metrics(y_test, y_preds)

        line = f"[{cat}->{cls}] Precision: {prc} Recall: {rcl} FBeta: {fb}"
        sliced_values.append(line)

with open('../data/sliced_output.txt', 'w') as f:
    for sliced_value in sliced_values:
        f.write(sliced_value + '\n')

# Testing
test.drop(columns=["salary"], inplace=True)
X_test, y_test, encoder, lb = process_data(
    test, encoder=encoder, lb=lb,
    categorical_features=cat_features, training=False)
print(X_test)
preds = inference(clf, X_test)
print(y_test)
print(preds)
y = lb.inverse_transform(preds)[0]
print(y)

# Save encoder and lb
joblib.dump(encoder, "../model/encoder.joblib")
joblib.dump(lb, "../model/lb.joblib")