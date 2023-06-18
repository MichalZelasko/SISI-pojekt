import requests
import base64


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def send_image_to_lambda(image_path):
    url = "https://hfiwxk6nnaznvxyalmzfjlmxqm0xutfs.lambda-url.us-east-1.on.aws/"
    # url = "http://localhost:8080/2015-03-31/functions/function/invocations"  # Local container

    headers = {'Content-Type': 'application/json'}
    payload = {
        'image': encode_image(image_path)
    }
    response = requests.post(url, json=payload, headers=headers)

    return response.text


image_path = 'normal.jpeg'
response = send_image_to_lambda(image_path)
print(response)
