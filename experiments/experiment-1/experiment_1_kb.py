import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
import pathlib
import sh
from models.wikitext_lm import WikitextLM

# !wandb login e279feeab3d602ab530e4eb23df8ac3ff3763461
# import IPython ; IPython.embed() ; exit(1)
data_dir = pathlib.Path(__file__).parent.absolute()
data_file = data_dir / 'data.txt'
# try:
#     sh.rm(data_dir / 'data.txt')
# except:
#     pass
run_params = {
        'model_name' : 'gpt2',
        'epochs' : 10,
        'debug' : False,
        'batch_size' : 8,
        'percent' : 10,
        'dataset' : 'wikitext',
        'seq_length' : 32,
        'statement_length' : 16,
        'momentum' : .9,
        'lr' : 1e-2,
        'repo' : 'wikitext-103-raw-v1',
        'data_file' : data_file,
}
model = WikitextLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-kb', project='experiment-1'),
)

trainer.fit(model)
