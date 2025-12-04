import optuna
import torch
from src.training.config_schema import TrainingConfig
from src.training.trainer import FluxTrainer
from src.common.logger import configure_logger

log = configure_logger("hyperparam_optimizer")

class HyperparamOptimizer:
    def __init__(self, n_trials=5):
        self.n_trials = n_trials

    def objective(self, trial, train_loader, val_loader):
        # 1. Sample Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        arch = trial.suggest_categorical("model_architecture", ["unet", "swin_unetr"])
        gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)
        
        # 2. Configure Run
        config = TrainingConfig(
            experiment_name="flux_optuna_sweep",
            run_name=f"trial_{trial.number}",
            learning_rate=lr,
            model_architecture=arch,
            focal_gamma=gamma,
            max_epochs=3 # Short epochs for tuning
        )
        
        # 3. Train
        trainer = FluxTrainer(config)
        final_dice = trainer.run_pipeline(train_loader, val_loader)
        
        return final_dice

    def run_optimization(self, train_loader, val_loader):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self.objective(t, train_loader, val_loader), n_trials=self.n_trials)
        
        log.info("tuning_complete")
        log.info("best_params", params=study.best_params)
        log.info("best_value", dice=study.best_value)
        return study.best_params