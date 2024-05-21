from typing import Dict
import wandb
import os


class WandBLogger:
    def __init__(
        self, wandb_api_key: str, run_config: Dict, run_name: str, group_name: str
    ):
        self.config = run_config
        self.run_name = run_name
        self.group_name = group_name
        wandb.login(key=wandb_api_key)
        self.run = wandb.init(
            project="lodnelf",
            name=run_name,
            config=run_config,
            group=group_name,
        )

    @classmethod
    def from_env(cls, run_config: Dict, run_name: str, group_name: str):
        wandb_api_key: str = str(os.getenv("WANDB_API_KEY"))
        if wandb_api_key is None:
            raise ValueError("WANDB_API_KEY environment variable is not set.")
        return cls(wandb_api_key, run_config, run_name, group_name)

    def log(self, metrics, step):
        wandb.log(metrics, step=step, commit=False)

    def commit(self):
        wandb.log({}, commit=True)

    def finish(self):
        self.run.finish()
