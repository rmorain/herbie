import requests
from rake_nltk import Rake

class WikidataClient():
    def __init__(self):
        self.rake = Rake()

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

    def _verify_wikidata_entity(self, entity):
        """
        Returns a wikidata entity id and checks that it's valid. Otherwise, returns None
        """
        assert entity.status_code == requests.codes.ok, "Request failed"
        try:
            entity_id = entity.json()['search'][0]['id']    # Index to get entity id from resource
        except:
            entity_id = None
        return entity_id

    def get_wikidata_entity_id(self, token):
        """
        Returns a string that points to a wikidata entity.

        Args:
            token (str):
                The token or phrase used for the search 
        """
        request = self._preprocess_wikidata_id_request(token)
        wikidata_entity = requests.get(request)
        entity_id = self._verify_wikidata_entity(wikidata_entity)
        return entity_id

    def _get_statement(self, x):
        """
        returns 
        """

    def extract_knowledge(self, x):
        """
        Modfies a datapoint, x, to include a new feature,
        'statement'. A 'statement' is a (str) that is formated
        subject:description.
        """
        try:
            self.rake.extract_keywords_from_text(x['text'])
            ranked_phrases = self.rake.get_ranked_phrases()
            for phrase in ranked_phrases:
                try:
                    id = self.get_wikidata_entity_id(phrase)
                    entity = self.client.get(id)
                    description = entity.attributes['descriptions']['en']['value']
                    break
                except:
                    continue        
            label = entity.attributes['labels']['en']['value']
            statement = label + ":" + description
            x['statement'] = statement
        except:
            x['statement'] = ""
        return x