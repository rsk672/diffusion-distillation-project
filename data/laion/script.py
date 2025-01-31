from datasets import load_dataset
from PIL import Image
import requests
import os
from io import BytesIO

image_save_directory = "./images"
text_output_file = "./caption.txt"

os.makedirs(image_save_directory, exist_ok=True)

cnt = 0

os.makedirs(image_save_directory, exist_ok=True)

with open(text_output_file, 'w', encoding='utf-8') as text_file:
    dataset = load_dataset("laion/aesthetics_v2_4.75", "default", streaming=True)

    for entry in dataset["train"]:
        url = entry['URL']
        text = entry['TEXT']
        width = entry['WIDTH']
        height = entry['HEIGHT']
        
        if not url or not text or not width or not height:
            continue

        if width < 512 or height < 512:
            continue

        try:
            response = requests.get(url, timeout=1)
            image = Image.open(BytesIO(response.content))
            actual_width, actual_height = image.size

            if actual_width >= 512 and actual_height >= 512:
                image_filename = os.path.join(image_save_directory, f'{cnt}.jpg')

                image.save(image_filename)

                text_file.write(f"{text}\n")
                cnt += 1
                
                if cnt >= 30000:
                    break

        except Exception as e:
            print(f"An error occurred while processing the image from {url}: {str(e)}")