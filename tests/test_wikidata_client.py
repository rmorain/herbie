import unittest
from models.utils.wikidata_client import WikidataClient

class TestWikidataClient(unittest.TestCase):
    
    def test_get_wikidata_entity_id(self):
        wikidata_client = WikidataClient()
        token = 'Elon Musk'
        result = wikidata_client.get_wikidata_entity_id(token)
        self.assertEqual(result, 'Q317521')

