import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, confusion_matrix
from lime.lime_tabular import LimeTabularExplainer

# Add this function to visualize feature importances
def plot_feature_importance(feature_importance, model_name):
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.title(f'Feature Importances for {model_name}')
    plt.show()



def convert_datetime_to_unix(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col]).astype(np.int64) // 10**9
    return df


def non_textual_gbm_model(project_name):

    predictors = ["created_date","updated_date",
                   "author_time_date",
                  "commit_time_date"]

    response_col = "label"

    train_df = pd.read_parquet('train_isis.parquet')
    test_df = pd.read_parquet('test_isis.parquet')

    # Convert datetime columns to Unix timestamps
    datetime_columns = ["created_date", "updated_date","commit_time_date", "author_time_date"]
    train_df = convert_datetime_to_unix(train_df, datetime_columns)
    test_df = convert_datetime_to_unix(test_df, datetime_columns)

    X_train = train_df[predictors].values
    y_train = train_df[response_col].values
    X_test = test_df[predictors].values
    y_test = test_df[response_col].values

    clf = GradientBoostingClassifier(n_estimators=60, max_depth=15, min_samples_split=2,
                                     learning_rate=0.01, random_state=1)
    clf.fit(X_train, y_train)

    # Visualize feature importances
    feature_importance = pd.DataFrame({'feature': predictors, 'importance': clf.feature_importances_})
    plot_feature_importance(feature_importance, 'GBM Model')

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Maximum F1 score:", f1)
    print("Maximum recall score:", recall)
    print("Maximum precision score:", precision)

    # Add confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # LIME integration
    explainer = LimeTabularExplainer(X_train, feature_names=predictors, class_names=['0', '1'], discretize_continuous=True)

    # Select a sample to explain
    sample_idx = 10
    sample = X_test[sample_idx].reshape(1, -1)

    # Explain the prediction
    exp = explainer.explain_instance(sample[0], clf.predict_proba, num_features=len(predictors))

    # Print the explanation
    print('\nSample Prediction Explanation:')
    exp.show_in_notebook()

    return y_pred

project_name = 'isis'
non_textual_gbm_pred = non_textual_gbm_model(project_name=project_name)

