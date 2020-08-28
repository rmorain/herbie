import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
from models.utils.wikidata_client import WikidataClient

class TestWikidataClient(unittest.TestCase):
    
    def test_get_wikidata_entity_id(self):
        # Check if a valid example works
        wikidata_client = WikidataClient()
        token = 'Elon Musk'
        result = wikidata_client.get_wikidata_entity_id(token)
        print(result)
        self.assertEqual(result, 'Q317521')

        # Check if a valid example returns None
        token = 'Robert Morain'
        result = wikidata_client.get_wikidata_entity_id(token)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()