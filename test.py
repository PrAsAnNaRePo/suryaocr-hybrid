import base64
import time

# use any one of it
from batch_suryaocr_llm import SuryaOCRAgent
# from suryaocr_gpt import SuryaOCRAgent
# from suryaocr_llm import SuryaOCRAgent

img_pah = 'image.png'

def convert_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

agent = SuryaOCRAgent()
start_time = time.time()
agent(convert_to_base64(img_pah), "batch-7b-super_coool")
print(time.time() - start_time)