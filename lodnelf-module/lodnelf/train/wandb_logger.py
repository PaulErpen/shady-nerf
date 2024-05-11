from typing import Dict
import wandb


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

    def log(self, metrics, step):
        wandb.log(metrics, step=step, commit=True)

    def finish(self):
        self.run.finish()
