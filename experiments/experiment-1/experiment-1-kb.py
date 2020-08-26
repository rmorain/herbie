from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import pytorch_lightning as pl
import torch
import sh
import nlp
import wandb
from pytorch_lightning.loggers import WandbLogger
from rake_nltk import Rake
from wikidata.client import Client
import requests

# !wandb login e279feeab3d602ab530e4eb23df8ac3ff3763461
# import IPython ; IPython.embed() ; exit(1)

# Flags for parameters that can be specified from command line
model_name = 'gpt2'
epochs = 1
debug = False
batch_size = 8
percent = 1
dataset = 'wikitext'
seq_length = 32
statement_length = 16
momentum = .9
lr = 1e-2
repo = 'wikitext-103-raw-v1'


model = WikitextLM()
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=epochs,
    fast_dev_run=debug,
    logger=WandbLogger(save_dir='logs/', name='wikitext-no-kb', project='experiment-1'),
)

trainer.fit(model)
