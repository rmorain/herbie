import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
from models.utils.data_writer import DataWriter 

class TestDataWriter(unittest.TestCase):
    def test_write(self):
        file_path = 'tests/test.txt'
        data_writer = DataWriter(file_path)
        x = {'statement':'emi:a cute girl'}
        data_writer.write(x)
        data_writer.close()

        # read data back
        file_object = open(file_path, 'r')
        line = file_object.read().strip('\n')
        self.assertEqual(line, x['statement'])
        file_object.close()
        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
