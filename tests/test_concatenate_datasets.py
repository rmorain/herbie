import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
import nlp

class TestExperiment(unittest.TestCase):
    def test_concatenate_datasets(self):
        data1, data2 = {"text": ['rob', 'sux', 'bad']}, {"statement": ['rob:a guy who sux', 'sux:sucking', 'bad:not good']}
        dset1, dset2 = nlp.Dataset.from_dict(data1), nlp.Dataset.from_dict(data2)
        import IPython; IPython.embed(); exit(1)        
        
        pass        
        
if __name__ == '__main__':
    unittest.main()
