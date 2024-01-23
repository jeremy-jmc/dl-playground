import mlflow
# https://github.com/manuelgilm/mlflow_for_ml_dev/tree/master
# TODO: https://github.com/elizobee/pipeline_with_mlflow/blob/main/pipeline_with_mlflow.py
# https://medium.com/@avikumart_/machine-learning-experiment-tracking-using-mlflow-8bba10f8f475
# https://medium.com/grid-solutions/exploring-mlflow-ui-1eaf583749ba
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes
import pandas as pd
from typing import Any, Optional, Union
import os
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split
SEED = 42

def create_mlflow_experiment(
    experiment_name: str, artifact_location: str, tags: dict[str, Any]
) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location.

    Parameters:
    ----------
    experiment_name: str
        The name of the experiment to create.
    artifact_location: str
        The artifact location of the experiment to create.
    tags: dict[str,Any]
        The tags of the experiment to create.

    Returns:
    -------
    experiment_id: str
        The id of the created experiment.
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except Exception as e:
        print('\t', e)
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> mlflow.entities.Experiment:
    """
    Retrieve the mlflow experiment with the given id or name.

    Parameters:
    ----------
    experiment_id: str
        The id of the experiment to retrieve.
    experiment_name: str
        The name of the experiment to retrieve.

    Returns:
    -------
    experiment: mlflow.entities.Experiment
        The mlflow experiment with the given id or name.
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    return experiment


df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

target = 'target'

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    stratify=y)

experiment_id = create_mlflow_experiment(
    experiment_name="test", artifact_location="mlruns", tags={"data": "iris"}
)
experiment = get_mlflow_experiment(experiment_id=experiment_id)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("Creation timestamp: {}".format(experiment.creation_time))

print('\trun experiment')
with mlflow.start_run(run_name="testing", experiment_id=experiment.experiment_id) as run:
    # Your machine learning code goes here
    os.makedirs("run_artifacts", exist_ok=True)

    # log parameters
    mlflow.log_param("learning_rate", 0.01)
    parameters = {
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 100,
        "loss_function": "mse",
        "optimizer": "adam"
    }
    mlflow.log_params(parameters)

    # log metrics
    mlflow.log_metric("random_metric", 0.01)
    metrics = {
        "mse": 0.01,
        "mae": 0.01,
        "rmse": 0.01,
        "r2": 0.01
    }
    mlflow.log_metrics(metrics)
    
    # log artifacts
    with open("hello_world.txt", "w") as f:
        f.write("Hello World!")
    mlflow.log_artifact(local_path="hello_world.txt", artifact_path="text_files")

    # train model
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)


    # log classification report
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    with open("./run_artifacts/classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    classification_report_df = pd.DataFrame(classification_report_dict).transpose()
    classification_report_df.to_csv("./run_artifacts/classification_report.csv", index=True)
    
    # log artifacts from folder
    with open("./run_artifacts/text_file.txt", "w") as f:
        f.write("Hello!")
    mlflow.log_artifacts(local_dir="./run_artifacts", artifact_path="run_artifacts")

    # log the precision-recall curve
    fig_pr = plt.figure()
    pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Precision-Recall Curve")
    plt.legend()

    mlflow.log_figure(fig_pr, "metrics/precision_recall_curve.png")

    # log the ROC curve
    fig_roc = plt.figure()
    roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("ROC Curve")
    plt.legend()

    mlflow.log_figure(fig_roc, "metrics/roc_curve.png")

    # log the confusion matrix
    fig_cm = plt.figure()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.legend()

    mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")

    # log the model
    # infer signature
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])
    model_signature = infer_signature(model_input=X_train, model_output=y_pred)

    # log model 
    mlflow.sklearn.log_model(sk_model=rfc, artifact_path="random_forest_classifier", signature=model_signature)

    # print run info
    print("run_id: {}".format(run.info.run_id))
    print("experiment_id: {}".format(run.info.experiment_id))
    print("status: {}".format(run.info.status))
    print("start_time: {}".format(run.info.start_time))
    print("end_time: {}".format(run.info.end_time))
    print("lifecycle_stage: {}".format(run.info.lifecycle_stage))


"""

# * run without run_name
mlflow.start_run()
# Your machine learning code goes here
mlflow.log_param("learning_rate",0.01)
#end the mlflow run
mlflow.end_run()


with mlflow.start_run(run_name="mlflow_runs") as run:
    # Your machine learning code goes here
    mlflow.log_param("learning_rate",0.01)
    print("RUN ID ->", run.info.run_id)

    print(run.info)

"""
