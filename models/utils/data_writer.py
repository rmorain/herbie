import os.path
from os import path

class DataWriter():
    """
    Writes data extracted from Wikidata to txt file
    """
    def __init__(self, file_path=None):
        # If the file does not exist, create the file
        if not path.exists(file_path):
            self.file_object = open(file_path, 'w')
        else:
            # Otherwise, we won't write to the file
            self.file_object = None

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
        
