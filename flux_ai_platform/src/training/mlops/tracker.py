import mlflow
import os
from src.common.logger import configure_logger
from src.training.config_schema import TrainingConfig

log = configure_logger("mlops_tracker")

class MLFlowTracker:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.active_run = None
        
        # Setup Backend (Local or Remote)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./experiments/mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        
        log.info("mlflow_initialized", uri=tracking_uri, experiment=config.experiment_name)

    def start_run(self):
        self.active_run = mlflow.start_run(run_name=self.config.run_name)
        # Log all configuration parameters automatically
        mlflow.log_params(self.config.model_dump())
        log.info("mlflow_run_started", run_id=self.active_run.info.run_id)

    def log_metric(self, key: str, value: float, step: int = None):
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception:
            pass # Don't crash training if logging fails

    def log_artifact(self, local_path: str):
        try:
            mlflow.log_artifact(local_path)
        except Exception as e:
            log.error("artifact_log_failed", error=str(e))

    def end_run(self):
        mlflow.end_run()