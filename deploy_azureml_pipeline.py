import argparse

from azure.ai.ml import Input, MLClient, dsl, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data, Environment
from azure.identity import DefaultAzureCredential


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", type=str, help="azure subscription id wher the workspace is located")
    parser.add_argument("--resource_group_name", type=str, help="azure resource group name where the workspace is located")
    parser.add_argument("--workspace_name", type=str, help="azure machine learning workspace name")
    parser.add_argument("--mlflow_experiment_name", type=str, help="mlflow experiment name")
    parser.add_argument("--registered_model_name", type=str, help="name of registered model")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--job_owner", type=str, help="owner of the model")
    parser.add_argument("--project_name", type=str, help="project of the model")
    parser.add_argument("--business_unit", type=str, help="business unit of the model")

    args = parser.parse_args()


    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name,
    )

    heart_data = Data(
        name="heart_data_csv",
        path="https://raw.githubusercontent.com/HooverCz/ML-API/dev/heart.csv",
        type=AssetTypes.URI_FILE,
        description="Dataset for heart disease prediction",
        tags={"source_type": "web", "project": "heart-disease-prediction"},
        version="1.0.0",
    )

    heart_data = ml_client.data.create_or_update(heart_data)
    print(
        f"Dataset with name {heart_data.name} was registered to workspace, the dataset version is {heart_data.version}"
    )


    custom_env_name = "aml-scikit-learn"

    pipeline_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for heart disease classification pipeline",
        tags={"scikit-learn": "0.24.2"},
        conda_file="dependencies/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        version="0.1.1",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    print(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
    )

    data_prep_component = load_component(source="components/data_prep/data_prep.yml")
    train_component = load_component(source="components/train/train.yml")
    data_prep_component = ml_client.create_or_update(data_prep_component)
    train_component = ml_client.create_or_update(train_component)

    # the dsl decorator tells the sdk that we are defining an Azure ML pipeline



    @dsl.pipeline(
        compute="serverless",
        description="heart-disease-pipeline",
    )
    def hearts_disease_pipeline(
        pipeline_job_data_input,
        pipeline_job_test_train_ratio,
        pipeline_job_mlflow_experiment_name,
        pipeline_job_registered_model_name,
        pipeline_job_owner,
        pipeline_job_project,
        pipeline_job_business_unit,
    ):
        # using data_prep_function like a python call with its own inputs
        data_prep_job = data_prep_component(
            data=pipeline_job_data_input,
            test_train_ratio=pipeline_job_test_train_ratio,
            mlflow_experiment_name=pipeline_job_mlflow_experiment_name,
        )

        # using train_func like a python call with its own inputs
        train_job = train_component(
            train_data=data_prep_job.outputs.train_data,  # note: using outputs from previous step
            test_data=data_prep_job.outputs.test_data,  # note: using outputs from previous step
            mlflow_experiment_name=pipeline_job_mlflow_experiment_name,
            registered_model_name=pipeline_job_registered_model_name,
            owner=pipeline_job_owner,
            project=pipeline_job_project,
            business_unit=pipeline_job_business_unit,
        )

        # a pipeline returns a dictionary of outputs
        # keys will code for the pipeline output identifier
        return {
            "pipeline_job_train_data": data_prep_job.outputs.train_data,
            "pipeline_job_test_data": data_prep_job.outputs.test_data,
        }

    # Let's instantiate the pipeline with the parameters of our choice
    pipeline = hearts_disease_pipeline(
        pipeline_job_data_input=Input(type="uri_file", path=heart_data.path),
        pipeline_job_test_train_ratio=0.2,
        pipeline_job_mlflow_experiment_name=args.mlflow_experiment_name,
        pipeline_job_registered_model_name=args.registered_model_name,
        pipeline_job_owner=args.job_owner,
        pipeline_job_project=args.project_name,
        pipeline_job_business_unit=args.business_unit,
    )

    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name=args.mlflow_experiment_name,
    )

if __name__ == "__main__":
    main()
