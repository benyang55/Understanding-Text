import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

def textprocessor():

# Instantiates a client
    client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
    file_name = os.path.abspath('longerimage.png')

# Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

        image = types.Image(content=content)
        response = client.document_text_detection(image=image)
        document = response.full_text_annotation
#print(document.text)
    output = document.text

    file = open("output.txt","w")
    file.write(output)
    file.close()

    return output
