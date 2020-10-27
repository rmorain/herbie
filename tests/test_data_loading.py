import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.dataset = 'wikitext'
        self.repo = 'wikitext-103-raw-v1'
        self.split = 'train'
        self.batch_size = 8
    
    def test_data_loading(self):
        # Load data from `datasets`
        ds = load_dataset(self.dataset, self.repo, split=f'{self.split}[:{self.batch_size}%"]')
        import pdb; pdb.set_trace()

        
        
      
if __name__ == '__main__':
    unittest.main()
