import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
import pathlib
import sh
import filecmp
from models.wikitext_lm import WikitextLM

class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        data_dir = pathlib.Path(__file__).parent.absolute()
        file_object = data_dir / 'data.txt'
        try:
            sh.rm(file_object)
        except:
            pass
        run_params = {
        'model_name' : 'gpt2',
        'epochs' : 1,
        'debug' : True,
        'batch_size' : 8,
        'percent' : 1,
        'dataset' : 'wikitext',
        'seq_length' : 32,
        'statement_length' : 16,
        'momentum' : .9,
        'lr' : 1e-2,
        'repo' : 'wikitext-103-raw-v1',
        'data_dir' : data_dir,
        }
        model = WikitextLM(run_params)
        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if torch.cuda.is_available() else 0),
            max_epochs=run_params['epochs'],
            fast_dev_run=run_params['debug'],
        )

        trainer.fit(model)
        self.assertTrue(filecmp.cmp('test-data.txt', str(file_object)))
        try:
            sh.rm(file_object)
        except:
            pass

if __name__ == '__main__':
    unittest.main()
