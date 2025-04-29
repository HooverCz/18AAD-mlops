import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve



def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--mlflow_experiment_name", type=str, help="mlflow experiment name")
    parser.add_argument("--registered_model_name", type=str, help="name of registered model")
    parser.add_argument("--owner", type=str, help="owner of the model")
    parser.add_argument("--project", type=str, help="project of the model")
    parser.add_argument("--business_unit", type=str, help="business unit of the model")

    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Extracting the label column
    y_train = train_df.pop("output")
    y_test = test_df.pop("output")


    X_train = train_df.values
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")
    # Define the columns to be encoded and scaled
    cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
    con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

    # Define the preprocessor for categorical and continuous columns
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    continuous_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('con', continuous_transformer, con_cols)
    ])

    # Combine preprocessing with the classifier in a single pipeline
    lr_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression())
    ])


    # Define hyperparameter grid for grid search
    param_grid = {
        'clf__penalty' : ['l1', 'l2'],
        'clf__C' : np.logspace(-4, 4, 20),
        'clf__solver' : ['liblinear']
        }

    lr_grid_search = GridSearchCV(lr_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    model_name = args.registered_model_name
    model_type = "LogisticRegression"

    # Fit the grid search to the data
    mlflow.set_experiment(experiment_name=args.mlflow_experiment_name)
    with mlflow.start_run(run_name=model_type):
        lr_grid_search.fit(X_train, y_train)

        # Get the best model from the grid search
        best_model = lr_grid_search.best_estimator_

        # Predict using the best model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        # Calculate test metrics
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_true=y_test, y_pred=y_pred),
            "test_recall": recall_score(y_true=y_test, y_pred=y_pred),
            "test_f1": f1_score(y_true=y_test, y_pred=y_pred),
        }
        tags = {
            "owner": args.owner,
            "project": args.project,
            "business_unit": args.business_unit,
            "model_type": model_type,
        }

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
        fig = plt.figure(figsize=(6, 4))
        # Plot the diagonal 50% line
        plt.plot([0, 1], [0, 1], 'k--')
        # Plot the FPR and TPR achieved by our model
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig("ROC-Curve.png")

        # Log parameters and metrics to MLflow
        mlflow.log_params(lr_grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("ROC-Curve.png")
        mlflow.set_tags(tags)

        signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))

        # Register the final model with MLflow
        mlflow.sklearn.log_model(
            best_model,
            artifact_path=model_name,
            input_example=X_train.iloc[:1],
            signature=signature,
            )
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
            name=model_name,
            tags={**tags, **metrics},
        )

        print(f"The test accuracy score of is {metrics['test_accuracy']}")


if __name__ == "__main__":
    main()
