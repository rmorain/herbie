import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
import pathlib
import sh
from models.utils.wikidata_client import WikidataClient

class TestWikidataClient(unittest.TestCase):
    def setUp(self):
        self.data_file = pathlib.Path(__file__).parent.absolute() / 'test.txt'
        self.wikidata_client = WikidataClient(self.data_file)

    def tearDown(self):
        try:
            self.wikidata_file.data_writer.close()
            sh.rm(self.data_file)
        except:
            pass

    def test_get_wikidata_entity_id(self):
        # Check if a valid example can be requested from wikidata. 
        token = 'Elon Musk'
        result = self.wikidata_client._get_wikidata_entity_id(token)
        self.assertEqual(result, 'Q317521')

        # Check if a valid example returns None
        token = 'Robert Morain'
        result = self.wikidata_client._get_wikidata_entity_id(token)
        self.assertIsNone(result)

    def test_get_entity_from_ranked_phrases(self):
        # Test to make sure it works when it should
        ranked_phrases = ['Tesla', 'a pair of shoes', 'waterfalls']
        entity = self.wikidata_client._get_entity_from_ranked_phrases(ranked_phrases)
        self.assertEqual(entity.label, 'Nikola Tesla')
        self.assertEqual(entity.description, 'Serbian-American inventor')

        # Test when no entitties exist
        ranked_phrases = ['', 'rob morain']
        entity = self.wikidata_client._get_entity_from_ranked_phrases(ranked_phrases)
        self.assertIsNone(entity.entity)

    def test_extract_knowledge(self):
        # Positive example
        x = {'text':'Steph Curry is my favorite basketball player'}
        x = self.wikidata_client.extract_knowledge(x)
        self.assertEqual(x['statement'], 'Stephen Curry:American basketball player')

        # Negative example
        x = {'text':'no no no no nononoooo'}
        x = self.wikidata_client.extract_knowledge(x)
        self.assertEqual(x['statement'], '')

        # Check if data was written to the file.
        self.data_file = open(self.data_file, 'r')
        lines = self.data_file.readlines()
        self.data_file.close()
        self.assertEqual(lines, ['Stephen Curry:American basketball player\n','\n'])
            
    def test_extract_knowledge_with_saved_data(self):
        # Test if the wikidata client reads from txt file when it exists
        # Write to text file
        text = 'Stephen Curry:American basketball player\n'
        f = open(self.data_file, 'w')
        f.write(text)
        f.write('\n')
        f.close()

        # Run extract function test
        self.test_extract_knowledge()

if __name__ == '__main__':
    unittest.main()
