import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt




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

# function to create a classification model
def predict_model(dataset,predictors,response):
    # ensure the results are reproducible 
    seed = 50
    train_features, test_features, train_labels, test_labels = test_train(dataset,predictors,response)
    # encoding categorical features to numeric
    features_to_encode = train_features.columns[train_features.dtypes == object].tolist()
    col_trans = make_column_transformer(
        (OneHotEncoder(), features_to_encode),
        remainder="passthrough"
    )
    # defining the classifier
    rf_classifier = RandomForestClassifier(
        min_samples_leaf=50,
        n_estimators=150,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=seed,
        max_features='sqrt')

    pipe = make_pipeline(col_trans, rf_classifier)
    pipe.fit(train_features, train_labels)
    y_pred = pipe.predict(test_features)
    #Accuracy (fraction of correctly classified samples)
    a_score = accuracy_score(test_labels, y_pred)
    # Make probability predictions
    train_probs = pipe.predict_proba(train_features)[:, 1]
    probs = pipe.predict_proba(test_features)[:, 1]
    train_predictions = pipe.predict(train_features)
    train_auc=roc_auc_score(train_labels, train_probs)
    test_auc=roc_auc_score(test_labels, probs)

    dummies = pd.get_dummies(dataset[features_to_encode])
    res = pd.concat([dummies, dataset], axis=1)
    X_train_encoded = res.drop(features_to_encode, axis=1)
    feature_importances = list(zip(X_train_encoded, rf_classifier.feature_importances_))
    feature_importances_ranked = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked];
    feature_names_25 = [i[0] for i in feature_importances_ranked[:25]]
    y_ticks = np.arange(0, len(feature_names_25))
    x_axis = [i[1] for i in feature_importances_ranked[:25]]
    plt.figure(figsize=(10, 14))
    plt.barh(feature_names_25, x_axis)  # horizontal barplot
    plt.title('Random Forest Feature Importance (Top 25)',
              fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})
    plt.xlabel('Features', fontdict={'fontsize': 16})
    plt.show()

    return a_score,train_probs,probs,train_predictions,y_pred,train_auc,test_auc,test_labels,train_labels


def evaluate_model(dataset,predictors,response):

    y_pred=predict_model(dataset,predictors,response)[4]
    probs=predict_model(dataset,predictors,response)[2]
    train_predictions=predict_model(dataset,predictors,response)[3]
    train_probs=predict_model(dataset,predictors,response)[1]
    y_test=predict_model(dataset,predictors,response)[7]
    y_train=predict_model(dataset,predictors,response)[8]
    
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()}'
              f'Baseline: {round(baseline[metric], 2)}'
              f'Test: {round(results[metric], 2)}'
              f'Train: {round(train_results[metric], 2)}')
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();


def main():
        data = import_data()
        # remove useless columns
        del data["Facility_ID"]
        del data["Unnamed: 0"]
        # create a response variable based on infections reported column, if any infections reported response =1 if no response =0
        data["response"] = [0 if x == 0 else 1 for x in data["Infections_Reported"]]
        # remove infections reported variable because we do not want the model to cheat
        del data["Infections_Reported"]
        # define response and predictors
        response = "response"
        predictors = [col for col in data.columns if col != response]
        print(f"The accuracy of the model is {round(predict_model(data,predictors,response)[0], 3) * 100} %")
        print(f'Train ROC AUC Score: {predict_model(data,predictors,response)[5]}')
        print(f'Test ROC AUC  Score: {predict_model(data,predictors,response)[6]}')
        # get ROC curve anf model evaluation stats
        evaluate_model(data, predictors, response)



if __name__ == "__main__":
    sys.exit(main())
