import base64
from datetime import datetime

from img2table.document import Image as DocImage
from PIL import Image
from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe
import surya
import polars as pl
import typing
import pandas as pd
import base64
import io
from PIL import Image
import cv2
import numpy as np
import time
from surya.detection import DetectionPredictor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class SuryaOCR(OCRInstance):
    """
    DocTR instance
    """
    def __init__(self):
        """
        Initialization of EasyOCR instance
        """
        self.detection_predictor = DetectionPredictor()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


    def content(self, document: Document) -> typing.List["surya.schema.OCRResult"]:
        
        images = [Image.fromarray(img) for img in document.images]
        detec_prediction = self.detection_predictor(images)

        slice_map = []
        all_langs = []
        all_slices = []
        all_polygons = []
        all_bboxes = []

        for idx, (det_pred, image, lang) in enumerate(zip(detec_prediction, images, ['en'])):
            polygons = [p.polygon for p in det_pred.bboxes]
            slices = self.slice_polys_from_image(image, polygons)
            slice_map.append(len(slices))
            all_langs.extend([lang] * len(slices))
            all_slices.extend(slices)
            all_polygons.extend(polygons)
            all_bboxes.extend([i.bbox for i in det_pred.bboxes])

        all_base64_images = [self.convert_to_base64(i) for i in all_slices]

        ocr = self.get_recognition_batch(all_base64_images)
        
        print("num cells: ", len(ocr))

        result = []
        for idx, (pred, polygon, bbox) in enumerate(zip(ocr, all_polygons, all_bboxes)):
            result.append(
                {
                    'polygon': polygon,
                    'confidence': 1.0,
                    'text': pred,
                    'bbox': bbox
                }
            )

        d = {
            "status": 'complete',
            'pages': [
                {
                    'text_lines': result,
                    'language': ['en'],
                    'page': 1
                }
            ]
        }
        return d["pages"]

    def get_recognition_batch(self, images: list):
        messages_batch = []
        for img in images:
            messages = [
                {
                    'role': 'system',
                    'content': "You are text extractor, you have extradinary ocr skills and can extract multiple languages, low opacity texts and cluttered text.\nExtact only the text from ths image, just respond with the extracted text alone. if you can't extract anything, respond with ''.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{img}"},
                        {"type": "text", "text": "Extract only the text from this image."},
                    ],
                }
            ]
            messages_batch.append(messages)

        # Process all messages in batch
        text_batch = [self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ) for msg in messages_batch]

        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=text_batch,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side='left'
        )
        
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        # Process outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts

    def convert_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def slice_polys_from_image(self, image: Image.Image, polys):
        image_array = np.array(image, dtype=np.uint8)
        lines = []
        for idx, poly in enumerate(polys):
            lines.append(self.slice_and_pad_poly(image_array, poly))
        return lines
    
    def slice_and_pad_poly(self, image_array: np.array, coordinates):
        # Draw polygon onto mask
        coordinates = [(corner[0], corner[1]) for corner in coordinates]
        bbox = [min([x[0] for x in coordinates]), min([x[1] for x in coordinates]), max([x[0] for x in coordinates]), max([x[1] for x in coordinates])]

        # We mask out anything not in the polygon
        cropped_polygon = image_array[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
        coordinates = [(x - bbox[0], y - bbox[1]) for x, y in coordinates]

        # Pad the area outside the polygon with the pad value
        mask = np.zeros(cropped_polygon.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(coordinates)], 1)
        mask = np.stack([mask] * 3, axis=-1)

        cropped_polygon[mask == 0] = 255
        rectangle_image = Image.fromarray(cropped_polygon)

        return rectangle_image

    def to_ocr_dataframe(self, content: typing.List["surya.schema.OCRResult"]) -> OCRDataframe:
        list_elements = []
        for page_id, ocr_result in enumerate(content):
            line_id = 0
            for text_line in ocr_result['text_lines']:
                line_id += 1
                words = text_line['text'].split()
                bbox = text_line['bbox']
                
                # Calculate width per character for approximation
                line_width = bbox[2] - bbox[0]
                avg_char_width = line_width / max(1, len(text_line['text']))
                
                # Split into words with approximate positions
                x_start = bbox[0]
                for word in words:
                    word_width = len(word) * avg_char_width
                    dict_word = {
                        "page": page_id,
                        "class": "ocrx_word",
                        "id": f"word_{page_id + 1}_{line_id}_{len(list_elements)}",
                        "parent": f"line_{page_id + 1}_{line_id}",
                        "value": word,
                        "confidence": round(100 * text_line['confidence']),
                        "x1": int(x_start),
                        "y1": int(bbox[1]),
                        "x2": int(x_start + word_width),
                        "y2": int(bbox[3])
                    }
                    list_elements.append(dict_word)
                    x_start += word_width + avg_char_width  # Add space width

        return OCRDataframe(df=pl.DataFrame(list_elements)) if list_elements else None


class SuryaOCRAgent():
    def __init__(self) -> None:
        self.ocr = SuryaOCR()
    
    def decode_base64(self, base64_string):
        decoded_bytes = base64.b64decode(base64_string)
        return decoded_bytes
    
    def __call__(self, base64: str, user_prompt: str):
        base64_bytes = self.decode_base64(base64)
        doc = DocImage(base64_bytes)
        doc.to_xlsx(
            dest=f'{user_prompt}-result.xlsx',
            ocr=self.ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=True,
            min_confidence=50
        )
        df = pd.read_excel(f'{user_prompt}-result.xlsx')
        return df.to_string(index=False, header=False)
