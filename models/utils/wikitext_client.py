class WikitextClient():
    def __init__(self):
        pass

    def get_id(self, token):
        """Request a word from Wikidata"""
        assert isinstance(token, str), "Request token not a string"
        endpoint = "http://wikidata.org/w/api.php?"
        action = "action=wbsearchentities&"
        search = "search=" + token + "&"
        language = "language=en&"
        format = "format=json"
        request = endpoint + action + search + language + format
        resource = requests.get(request)
        assert resource.status_code == requests.codes.ok
        try:
            id = resource.json()['search'][0]['id']
        except:
            id = None
        return id