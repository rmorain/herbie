import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pathlib
import sh
from models.wikitext_kb_lm import WikitextKBLM
from models.wikitext_lm import WikitextLM

# Project
project = 'small_medium_large'

# Data
data_dir = pathlib.Path(srcdir + 'data')
data_file = data_dir / 'data25percent.txt'

# !wandb login e279feeab3d602ab530e4eb23df8ac3ff3763461

# Papa bear

# Without Knowledge Base

run_params = {
        'model_name' : 'gpt2',
        'max_epochs' : 50,
        'debug' : False,
        'batch_size' : 8,
        'percent' : 25,
        'dataset' : 'wikitext',
        'seq_length' : 32,
        'statement_length' : 16,
        'momentum' : .9,
        'lr' : 1e-2,
        'repo' : 'wikitext-103-raw-v1',
        'num_workers' : 8,
        'data_file' : data_file,
}

model = WikitextLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['max_epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-no-kb-large', project=project),
)

trainer.fit(model)

# With Knowledge Base

model = WikitextKBLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['max_epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-kb-large', project=project),
)

trainer.fit(model)

# Mama bear
run_params['percent'] = 10

# Without Knowledge Base

model = WikitextLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['max_epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-no-kb-medium', project=project),
)

trainer.fit(model)

# With Knowledge Base

model = WikitextKBLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['max_epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-kb-medium', project=project),
)

trainer.fit(model)

# Baby bear
run_params['percent'] = 5

# Without Knowledge Base

model = WikitextLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['max_epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-no-kb-small', project=project),
)

trainer.fit(model)

# With Knowledge Base

model = WikitextKBLM(run_params)
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=run_params['max_epochs'],
    fast_dev_run=run_params['debug'],
    logger=WandbLogger(save_dir='logs/', name='wikitext-kb-small', project=project),
)

trainer.fit(model)
