
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
import time
import requests
from openai import OpenAI
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

driver = webdriver.Chrome()

url = 'https://www.falabella.com.pe/falabella-pe/search?Ntt=puma+caven'
driver.get(url)
driver.maximize_window()

xpath_expression = '//*[@id="testId-searchResults-products"]/div[1]'
# '//*[@id="testId-searchResults-products"]/div'
# '//*[@id="testId-searchResults-products"]/div[1]//img/@src'

element = driver.find_element(By.XPATH, xpath_expression)
text = element.get_attribute('outerHTML')

print(text)

screenshot = element.screenshot_as_png

image = Image.open(io.BytesIO(screenshot))
plt.imshow(image)

image.save('screenshot.png')
time.sleep(5)

driver.quit()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"""
What is the product, current price, original price, discount and retailer of this product HTML description: {text}. 
If none of these attributes are available, return empty dict.
If any of these attributes are not available, return empty string in their respective key. 
Give me the result only as a Python dict. 
""",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion)
print(dir(chat_completion))
print('chat_completion to dict')
print(chat_completion.__dict__['choices'][0].__dict__['message'].content)

