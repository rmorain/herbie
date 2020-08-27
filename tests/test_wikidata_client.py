import unittest
from models.utils.wikidata_client import WikidataClient

class TestWikidataClient(unittest.TestCase):
    def __init__(self):
        self.wikidata_client = WikidataClient()
    
    def test_get_wikidata_entity_id(self):
        token = 'Elon Musk'
        result = self.wikidata_client.get_wikidata_entity_id(token)
        import IPython ; IPython.embed() ; exit(1)


