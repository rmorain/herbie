import requests
from rake_nltk import Rake
from wikidata.client import Client
from models.utils.data_writer import DataWriter
import pathlib
from nlp import load_dataset

class WikidataClient():
    """
    Contains 'extract_knowledge' method which extracts knowledge from
    the Wikidata knowledge base. All other submethods and subclasses exist
    to help the 'extract_knowledge' method.
    """
    def __init__(self, data_file):
        self.rake = Rake()
        self.client = Client()
        self.data_file = data_file 
        self.ds = None
        self.data_writer = None 
        self.reading = False
        if self.data_file.is_file():
            self.data_file = open(self.data_file, 'r')
            self.reading = True
        else:   
            # Creates a file to write to
            self.data_writer = DataWriter(self.data_file)
        
    
    class WikidataEntity():
        """
        Class that holds attributes related to a Wikidata entity

        attributes:
            id: (str) Identifies wikidata entity.
            label: (str) The name of the wikidata entity.
            description: (str) A description of the entity.
            entity: (Entity class) Wikidata entity object. Used to retrieve label and description.
        """
        def __init__(self):
            self.id = None  
            self.label = None
            self.description = None
            self.entity = None        

        def is_ok(self):
            if self.label and self.description:
                return True
            else:
                return False

    def extract_knowledge(self, x):
        """
        Modfies a datapoint, x, to include a new feature,
        'statement'. A 'statement' is a (str) that is formated
        subject:description.
        """
        # If data file is specified read from it
        if self.reading:
            line = self.data_file.readline()
            x['statement'] = line.strip('\n')
            return x
           
        wikidata_entity = self._get_wikidata_entity(x)
        if wikidata_entity.is_ok():
            statement = wikidata_entity.label + ":" + wikidata_entity.description
            x['statement'] = statement
        else:
            x['statement'] = ""
        self.data_writer.write(x)
        return x

    def close(self):
        """
        If there is a file to close it closes it
        """
        if self.data_writer:
            self.data_writer.close()
        if self.reading:
            self.data_file.close()

    def _get_wikidata_entity(self, x):
        """
        Returns a WikidataEntity object that contains all the information needed to create a statement
        """
        self.rake.extract_keywords_from_text(x['text'])
        ranked_phrases = self.rake.get_ranked_phrases()
        wikidata_entity = self._get_entity_from_ranked_phrases(ranked_phrases)
        return wikidata_entity

    def _get_entity_from_ranked_phrases(self, ranked_phrases):
        """
        Returns a wikidata entity object with as much information
        as possible.

        Args:
            ranked_phrases (list[str]): 
                Each phrase is used to 
        """
        entity = self.WikidataEntity()
        for phrase in ranked_phrases:
            try:
                entity.id = self._get_wikidata_entity_id(phrase)
                entity.entity = self.client.get(entity.id)
                entity.description = entity.entity.attributes['descriptions']['en']['value']
                entity.label = entity.entity.attributes['labels']['en']['value']
                break
            except:
                entity.__init__()  # clear variables
                continue
        return entity

    def _get_wikidata_entity_id(self, token):
        """
        Returns a string that points to a wikidata entity.

        Args:
            token (str):
                The token or phrase used for the search 
        """
        request = self._preprocess_wikidata_id_request(token)
        wikidata_entity = requests.get(request)
        entity_id = self._validate_wikidata_entity_id(wikidata_entity)
        return entity_id

    def _preprocess_wikidata_id_request(self, token):
        """
        Returns a string used to make wikidata request
        """
        assert isinstance(token, str), "Request token not a string"
        endpoint = "http://wikidata.org/w/api.php?"
        action = "action=wbsearchentities&"
        search = "search=" + token + "&"
        language = "language=en&"
        format = "format=json"
        return endpoint + action + search + language + format

    def _validate_wikidata_entity_id(self, entity):
        """
        Returns a wikidata entity id and checks that it's valid. Otherwise, returns None
        """
        assert entity.status_code == requests.codes.ok, "Request failed"
        try:
            entity_id = entity.json()['search'][0]['id']    # Index to get entity id from resource
        except:
            entity_id = None
        return entity_id
