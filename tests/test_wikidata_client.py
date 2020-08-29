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
        result = wikidata_client._get_wikidata_entity_id(token)
        self.assertEqual(result, 'Q317521')

        # Check if a valid example returns None
        token = 'Robert Morain'
        result = wikidata_client._get_wikidata_entity_id(token)
        self.assertIsNone(result)

    def test_get_entity_from_ranked_phrases(self):
        # Test to make sure it works when it should
        wikidata_client = WikidataClient()
        ranked_phrases = ['Tesla', 'a pair of shoes', 'waterfalls']
        entity = wikidata_client._get_entity_from_ranked_phrases(ranked_phrases)
        self.assertEqual(entity.label, 'Nikola Tesla')
        self.assertEqual(entity.description, 'Serbian-American inventor')

        # Test when no entitties exist
        ranked_phrases = ['', 'rob morain']
        entity = wikidata_client._get_entity_from_ranked_phrases(ranked_phrases)
        self.assertIsNone(entity.entity)


if __name__ == '__main__':
    unittest.main()