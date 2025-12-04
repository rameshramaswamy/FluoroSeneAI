import torch
import torch.optim as optim
import os
from src.training.config_schema import TrainingConfig
from src.training.models.factory import ModelFactory
from src.training.losses import get_loss_function
from src.training.mlops.tracker import MLFlowTracker
from src.common.logger import configure_logger
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureType

log = configure_logger("flux_trainer")

class FluxTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelFactory.create_model(config).to(self.device)
        self.loss_func = get_loss_function(config)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=1e-5
        )
        self.tracker = MLFlowTracker(config)

    def train_epoch(self, data_loader, epoch_idx):
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch_data in data_loader:
            steps += 1
            # MONAI DataLoader returns a dictionary
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # SwinUNETR returns hidden states tuple, we only want the first
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            # Calculate Loss
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Handle case where loader is empty
        if steps == 0: return 0.0

        avg_loss = total_loss / steps
        self.tracker.log_metric("train_loss", avg_loss, step=epoch_idx)
        return avg_loss

    def validate(self, data_loader, epoch_idx):
        self.model.eval()
        total_dice = 0.0
        steps = 0
        
        # Metric calculation for Medical Segmentation
        from monai.metrics import DiceMetric
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        with torch.no_grad():
            for batch_data in data_loader:
                steps += 1
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]

                # Convert logits to One-Hot Discrete for metric calculation
                # (Assuming binary segmentation for simplicity here)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)
                
                # Update Metric
                # Note: Labels might need One-Hot conversion depending on format
                dice_metric(y_pred=preds, y=labels)

            # Aggregate
            if steps > 0:
                score = dice_metric.aggregate().item()
                dice_metric.reset()
            else:
                score = 0.0

        self.tracker.log_metric("val_dice", score, step=epoch_idx)
        return score

    def run_pipeline(self, train_loader, val_loader):
        log.info("starting_training_pipeline", epochs=self.config.max_epochs)
        self.tracker.start_run()
        
        best_dice = 0.0
        
        try:
            for epoch in range(self.config.max_epochs):
                loss = self.train_epoch(train_loader, epoch)
                dice = self.validate(val_loader, epoch)
                
                log.info("epoch_finished", epoch=epoch, loss=f"{loss:.4f}", dice=f"{dice:.4f}")
                
                # Model Checkpointing
                if dice > best_dice:
                    best_dice = dice
                    save_path = f"best_model_{self.config.model_architecture}.pt"
                    torch.save(self.model.state_dict(), save_path)
                    self.tracker.log_artifact(save_path)
                    log.info("new_champion_saved", dice=dice)

        except Exception as e:
            log.error("training_crashed", error=str(e))
            raise e
        finally:
            self.tracker.end_run()
            
        return best_dice