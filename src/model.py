import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score




def import_data():
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, "../data/SSI_data_preprocessed.csv")
    data = pd.read_csv(
        data_path )
    return data


def test_train(dataset,predictors,response):
    # RF model
    # defining features dataset without labels
    features = dataset[predictors]
    # Labels are the values we want to predict
    labels = dataset[response]
    # Split the data into training and testing sets
    seed = 50
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=0
    )
    return train_features, test_features, train_labels, test_labels


def predict_model(dataset,predictors,response):
    seed = 50
    train_features, test_features, train_labels, test_labels = test_train(dataset,predictors,response)
    features_to_encode = train_features.columns[train_features.dtypes == object].tolist()
    col_trans = make_column_transformer(
        (OneHotEncoder(), features_to_encode),
        remainder="passthrough"
    )
    rf_classifier = RandomForestClassifier(
        min_samples_leaf=50,
        n_estimators=150,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=seed,
        max_features='auto')

    pipe = make_pipeline(col_trans, rf_classifier)
    pipe.fit(train_features, train_labels)
    y_pred = pipe.predict(test_features)
    accuracy_score(test_labels, y_pred)
    print(f"The accuracy of the model is {round(accuracy_score(test_labels, y_pred), 3) * 100} %")
    train_probs = pipe.predict_proba(train_features)[:, 1]
    probs = pipe.predict_proba(test_features)[:, 1]
    train_predictions = pipe.predict(train_features)
    print(f'Train ROC AUC Score: {roc_auc_score(train_labels, train_probs)}')
    print(f'Test ROC AUC  Score: {roc_auc_score(test_labels, probs)}')



def main():
        data = import_data()
        del data["Facility_ID"]
        del data["Unnamed: 0"]
        data["response"] = [0 if x == 0 else 1 for x in data["Infections_Reported"]]
        del data["Infections_Reported"]
        response = "response"
        predictors = [col for col in data.columns if col != response]
        predict_model(data,predictors,response)



if __name__ == "__main__":
    sys.exit(main())
