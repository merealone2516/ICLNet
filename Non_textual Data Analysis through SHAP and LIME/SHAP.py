import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, confusion_matrix
import shap

def non_textual_gbm_model(project_name):
    train_path = '/content/freemarker/train/'
    test_path = '/content/freemarker/test/'

    predictors = ["created_date","updated_date", "commit_time_date", "author_time_date", 
                  "bug","new feature", "task", "closed", "open", "resolved"]
    response_col = "label"

    train_df = pd.read_parquet('train_isis.parquet')
    test_df = pd.read_parquet('test_isis.parquet')

    # Convert datetime to ordinal
    for date_col in ['created_date', 'updated_date', 'commit_time_date', 'author_time_date']:
       train_df[date_col] = train_df[date_col].apply(lambda x: x.toordinal())
       test_df[date_col] = test_df[date_col].apply(lambda x: x.toordinal())

    # Train a GradientBoostingClassifier
    my_gbm = GradientBoostingClassifier(n_estimators=60, max_depth=15, min_samples_split=2, learning_rate=0.01, random_state=1)
    my_gbm.fit(train_df[predictors], train_df[response_col])

    # Compute and visualize SHAP values
    explainer = shap.TreeExplainer(my_gbm)
    shap_values = explainer.shap_values(train_df[predictors])
    shap.summary_plot(shap_values, train_df[predictors])

    y_pred = my_gbm.predict(test_df[predictors])
    y_true = test_df[response_col]

    report = classification_report(y_true, y_pred)
    print(report)

    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print("Maximum F1 score:", f1)
    print("Maximum recall score:", recall)
    print("Maximum precision score:", precision)

    pred = pd.DataFrame(y_pred, columns=['predict'])
    pred.to_csv("pred.csv")

    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return pred.to_numpy()

project_name='isis'
non_textual_gbm_pred = non_textual_gbm_model(project_name=project_name)
