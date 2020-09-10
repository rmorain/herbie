import os.path
from os import path

class DataWriter():
    """
    Writes data extracted from Wikidata to txt file
    """
    def __init__(self, file_path):
        # If the file does not exist, create the file
        self.file_object = open(file_path, 'w')

    def write(self, x):
        """
            Write statement to a textfile 
        """
        if self.file_object:
            self.file_object.write(x['statement'])
            self.file_object.write('\n')

    def close(self):
        if self.file_object:
            self.file_object.close()
        
