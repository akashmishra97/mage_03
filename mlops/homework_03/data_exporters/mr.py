if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    import mlflow
    import mlflow.sklearn
    from tempfile import TemporaryDirectory
    import os
    import pickle

    # Unpack the data
    dv = data['dict_vectorizer']
    model = data['model']

    # Log the model and the dict vectorizer with MLflow
    with mlflow.start_run() as run:
        # Log the linear regression model
        mlflow.sklearn.log_model(model, "linear_regression_model")
        
        # Save the dict vectorizer to a temporary directory and log it as an artifact
        with TemporaryDirectory() as temp_dir:
            dv_path = os.path.join(temp_dir, "dict_vectorizer.pkl")
            with open(dv_path, 'wb') as f:
                pickle.dump(dv, f)
            mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")

        # Create a new directory for the MLflow project
        new_project_dir = "/home/mlflow/MyNewProject"
        os.makedirs(new_project_dir, exist_ok=True)

        # Define the content of the MLproject file
        mlproject_content = """
        name: MyNewProject
        conda_env: conda.yaml
        """

        # Write the MLproject content to a file
        mlproject_file = os.path.join(new_project_dir, "MLproject")
        with open(mlproject_file, "w") as f:
            f.write(mlproject_content)
    
    # Return the run ID for reference
    return {'run_id': run.info.run_id}
