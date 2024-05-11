from typing import Any, Callable, Dict
from lodnelf.train.train_executor import TrainExecutor
from pathlib import Path
import torch
from lodnelf.train.wandb_logger import WandBLogger


class TrainHandler:
    def __init__(
        self,
        max_epochs: int,
        dataset,
        train_executor: TrainExecutor,
        train_config: Dict,
        group_name: str,
        prepare_input_fn: Callable[[Any], Any] | None = None,
        stop_after_no_improvement: int = 3,
        model_save_path: Path | None = None,
    ):
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.train_executor = train_executor
        self.prepare_input_fn = prepare_input_fn
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

        best_loss = float("inf")
        no_improvement = 0
        for epoch in range(self.max_epochs):
            current_loss = self.train_executor.train(
                self.dataset, prepare_input_fn=self.prepare_input_fn
            )
            print(f"Epoch {epoch} finished.")
            print(f"Current loss: {current_loss}")
            logger.log({"loss": current_loss}, step=epoch)
            if current_loss < best_loss:
                best_loss = current_loss
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

        print("Training finished.")
        logger.finish()
