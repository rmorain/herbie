import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class Experiment():
    def __init__(self, model, run_params):
        """
            model: Huggingface model
            run_params: Type RunParams
        """
        self.model = model(run_params)
        self.run_params = run_params

    def run(self):
        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if torch.cuda.is_available() else 0),
            max_epochs=self.run_params.max_epochs,
            fast_dev_run=self.run_params.debug,
            logger=WandbLogger(save_dir='logs/', 
                                name=self.run_params.run_name, 
                                project=self.run_params.project_name),
        )

        trainer.fit(model)
