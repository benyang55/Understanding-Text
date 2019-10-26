import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
#GOOGLE_APPLICATION_CREDENTIALS = /Users/benjaminyang/Desktop/CalHacks/My/Project/35981-21cbc1be1c57.json
# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('testimage2.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

#image = types.Image(content=content)

# Performs label detection on the image file
#response = client.label_detection(image=image)
#labels = response.label_annotations

#print('Labels:')
#for label in labels:
    #print(label.description)

#print(response)


image = types.Image(content=content)
response = client.document_text_detection(image=image)
document = response.full_text_annotation

return document.text

#print(document.text)
