import mlflow

# Configure MLflow tracking URI (can be local or remote server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Resume Ranking Experiment")

def log_metrics(similarity_scores):
    with mlflow.start_run():
        mlflow.log_metric("max_score", max(similarity_scores))
        mlflow.log_metric("min_score", min(similarity_scores))
        mlflow.log_metric("average_score", sum(similarity_scores) / len(similarity_scores))
