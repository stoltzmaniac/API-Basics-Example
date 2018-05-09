import json
import requests

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


def microsoft(image_url):
    # https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts/python
    creds = json.load(open('microsoft-credentials.json'))
    subscription_key = creds['subscription_id']
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    params = {'visualFeatures': 'Categories,Tags,Description,Faces,ImageType,Color,Adult',
              'details': 'Celebrities,Landmarks'}
    vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
    vision_analyze_url = vision_base_url + "analyze"
    data = {'url': image_url}
    response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    analysis = response.json()
    return analysis


def google(image_url, detection_type='label_detection'):
    # https://googlecloudplatform.github.io/google-cloud-python/latest/vision/index.html
    client = vision.ImageAnnotatorClient()
    image = types.Image()
    image.source.image_uri = image_url
    if detection_type == 'web_detection':
        response = client.web_detection(image=image).web_detection
    elif detection_type == 'label_detection':
        response = client.label_detection(image=image).label_annotations
    return response
