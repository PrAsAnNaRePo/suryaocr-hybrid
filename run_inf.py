import base64
import os
import time
from tqdm import tqdm

# from suryaocr_llm import SuryaOCRAgent
from batch_suryaocr_llm import SuryaOCRAgent

dir_path = 'cropped_images'

def convert_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

agent = SuryaOCRAgent()

print("TOTAL IMAGES: ", len(os.listdir(dir_path)))

start_time = time.time()
for img in tqdm(os.listdir(dir_path)):
    img_path = dir_path + '/' + img
    agent(convert_to_base64(img_path), img)
print(time.time() - start_time)

