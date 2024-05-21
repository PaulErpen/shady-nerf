from typing import Dict
from lodnelf.train.train_executor import TrainExecutor
from pathlib import Path
import torch
from lodnelf.train.wandb_logger import WandBLogger
from lodnelf.train.validation_executor import ValidationExecutor


class TrainHandler:
    def __init__(
        self,
        max_epochs: int,
        train_executor: TrainExecutor,
        validation_executor: ValidationExecutor,
        train_config: Dict,
        group_name: str,
        stop_after_no_improvement: int = 3,
        validation_frequency: int = 1,
        model_save_path: Path | None = None,
    ):
        self.max_epochs = max_epochs
        self.train_executor = train_executor
        self.validation_executor = validation_executor
        self.validation_frequency = validation_frequency
        self.stop_after_no_improvement = stop_after_no_improvement
        self.model_save_path = model_save_path
        self.train_config = train_config
        self.group_name = group_name

    def run(self, run_name: str):
        logger = WandBLogger.from_env(
            run_config=self.train_config,
            run_name=run_name,
            group_name=self.group_name,
        )

        best_validation_loss = float("inf")
        no_improvement = 0

        for epoch in range(self.max_epochs):
            train_loss = self.train_executor.train()
            print(f"Train loss: {train_loss}")
            logger.log({"train_loss": train_loss}, step=epoch)

            if epoch % self.validation_frequency == 0:
                val_loss = self.validation_executor.validate()
                print(f"Validation loss: {val_loss}")
                logger.log({"validation_loss": val_loss}, step=epoch)
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    no_improvement = 0
                    if self.model_save_path is not None:
                        torch.save(
                            self.train_executor.model.state_dict(),
                            self.model_save_path / f"model_epoch_{epoch}.pt",
                        )
                else:
                    no_improvement += 1
                    if no_improvement >= self.stop_after_no_improvement:
                        print(
                            f"No improvement in the last {no_improvement} epochs. Stopping training."
                        )
                        break
            print(f"Epoch {epoch} finished.")

        print("Training finished.")
        logger.finish()
