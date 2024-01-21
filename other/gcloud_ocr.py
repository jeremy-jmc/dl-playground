# https://beranger.medium.com/detect-text-on-image-using-google-cloud-vision-api-python-44c6e7430c44
import json
import subprocess
import requests

file_path = "./img/prueba_falabella.png"

command = f"gcloud ml vision detect-text {file_path} --language-hints=es"
result = subprocess.run(command, shell=True, capture_output=True, text=True)

print(result)

stdout_dict = json.loads(result.stdout)
print(stdout_dict)

print(stdout_dict['responses'])
print(len(stdout_dict['responses']))

print(json.dumps(stdout_dict['responses'][0]['fullTextAnnotation'], indent=2))

ocr_text = stdout_dict['responses'][0]['fullTextAnnotation']['text']
print(ocr_text)

