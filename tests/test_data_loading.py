import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
from models.utils.prepare_data import load_data
from models.utils.run_params import RunParams

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.run_params = RunParams() 
        
      
if __name__ == '__main__':
    unittest.main()
