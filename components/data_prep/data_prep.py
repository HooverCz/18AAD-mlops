import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow


# input and output arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to input data")
parser.add_argument("--mlflow_experiment_name", type=str, help="mlflow experiment name")
parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
parser.add_argument("--train_data", type=str, help="path to train data")
parser.add_argument("--test_data", type=str, help="path to test data")
args = parser.parse_args()

print(args)

# Start Logging
mlflow.set_experiment(experiment_name=args.mlflow_experiment_name)
with mlflow.start_run(run_name="DataPrep"):

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    df = pd.read_csv(args.data)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", df.shape[1] - 1)

    # Split data and train/test the pipeline
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_train_ratio,
    )

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df_train.to_csv(os.path.join(args.train_data, "data.csv"), index=False)

    df_test.to_csv(os.path.join(args.test_data, "data.csv"), index=False)
