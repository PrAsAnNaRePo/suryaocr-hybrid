import base64
from img2table.document import Image as DocImage
from PIL import Image
from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe
import surya
import polars as pl
import typing
import base64
import io
from typing import List
from PIL import Image
import cv2
import numpy as np
import time
from surya.detection import DetectionPredictor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
import torch

class SuryaOCR(OCRInstance):
    def __init__(self, max_height=1500, min_height=350, padding=4):
        """
        Initialization of EasyOCR instance with image scaling parameters
        
        Args:
            max_height (int): Maximum height for images during detection
            min_height (int): Minimum height for images during detection
            padding (int): Number of pixels to add around each bbox
        """
        self.detection_predictor = DetectionPredictor(device="cuda")
        self.max_height = max_height
        self.min_height = min_height
        self.padding = padding
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
            print("Loaded model successfully")
        except Exception as e:
            print("Failed to load model:", str(e))

    def scale_image_if_needed(self, image: Image.Image, min_height=300) -> tuple[Image.Image, float]:
        """
        Scale image down if height exceeds max_height or scale up if height is below min_height
        
        Args:
            image: PIL Image to scale
            min_height: Minimum height required for detection (default 300)
            
        Returns:
            tuple: (scaled image, scale factor)
        """
        # Case 1: Image is too small - scale UP
        if image.height < min_height:
            scale_factor = min_height / image.height
            new_width = int(image.width * scale_factor)
            scaled_image = image.resize((new_width, min_height), Image.Resampling.LANCZOS)
            print(f"Image scaled UP by factor {scale_factor:.2f} (height: {image.height} -> {scaled_image.height})")
            return scaled_image, scale_factor
        
        # Case 2: Image is too large - scale DOWN
        elif image.height > self.max_height:
            scale_factor = self.max_height / image.height
            new_width = int(image.width * scale_factor)
            scaled_image = image.resize((new_width, self.max_height), Image.Resampling.LANCZOS)
            print(f"Image scaled DOWN by factor {scale_factor:.2f} (height: {image.height} -> {scaled_image.height})")
            return scaled_image, scale_factor
        
        # Case 3: Image is within acceptable range - no scaling needed
        else:
            print(f"Image not scaled (height: {image.height} within range {min_height}-{self.max_height})")
            return image, 1.0

    def pad_bbox(self, bbox: List[int], image_width: int, image_height: int) -> List[int]:
        """
        Add padding to bbox while ensuring it stays within image boundaries
        
        Args:
            bbox: List of [x1, y1, x2, y2] coordinates
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            List of padded coordinates
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(image_width, x2 + self.padding)
        y2 = min(image_height, y2 + self.padding)
        return [x1, y1, x2, y2]

    def pad_polygon(self, polygon: List[List[int]], image_width: int, image_height: int) -> List[List[int]]:
        """
        Add padding to polygon while ensuring it stays within image boundaries
        
        Args:
            polygon: List of [x, y] coordinates
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            List of padded coordinates
        """
        # Get bounding box of polygon
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate padding ratios
        dx = self.padding / (max_x - min_x) if max_x != min_x else 0
        dy = self.padding / (max_y - min_y) if max_y != min_y else 0
        
        # Apply padding to each point
        padded_polygon = []
        for x, y in polygon:
            # Scale point outward from center
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            new_x = cx + (x - cx) * (1 + dx)
            new_y = cy + (y - cy) * (1 + dy)
            
            # Clip to image boundaries
            new_x = max(0, min(image_width, new_x))
            new_y = max(0, min(image_height, new_y))
            padded_polygon.append([int(new_x), int(new_y)])
            
        return padded_polygon

    def scale_bbox(self, bbox: List[int], scale_factor: float) -> List[int]:
        """
        Scale bounding box coordinates back to original image size
        
        Args:
            bbox: List of [x1, y1, x2, y2] coordinates
            scale_factor: Factor to scale coordinates by
                (>1.0 means image was scaled up, <1.0 means image was scaled down)
            
        Returns:
            List of scaled coordinates
        """
        # If scale_factor is 1.0, no scaling is needed
        if scale_factor == 1.0:
            return bbox
        
        # Divide by scale_factor to revert to original coordinates
        return [int(coord / scale_factor) for coord in bbox]

    def scale_polygon(self, polygon: List[List[int]], scale_factor: float) -> List[List[int]]:
        """
        Scale polygon coordinates back to original image size
        
        Args:
            polygon: List of [x, y] coordinates
            scale_factor: Factor to scale coordinates by
            
        Returns:
            List of scaled coordinates
        """
        # If scale_factor is 1.0, no scaling is needed
        if scale_factor == 1.0:
            return polygon
        return [[int(x / scale_factor), int(y / scale_factor)] for x, y in polygon]

    def content(self, document: Document) -> typing.List["surya.schema.OCRResult"]:
        valid_images = []
        scale_factors = []
        scaled_images = []
        
        # Validate and scale images if needed
        for idx, img in enumerate(document.images):
            if img is None:
                print(f"Warning: Image at index {idx} is None, skipping.")
                continue
            if hasattr(img, "size") and (img.size == 0):
                print(f"Warning: Image at index {idx} is empty (size={img.size}), skipping.")
                continue
            
            try:
                pil_img = Image.fromarray(img)
                valid_images.append(pil_img)
                
                # Scale image only if height exceeds max_height
                scaled_img, scale_factor = self.scale_image_if_needed(pil_img)
                scaled_images.append(scaled_img)
                scale_factors.append(scale_factor)
                
                if scale_factor < 1.0:
                    print(f"Image {idx} scaled down by factor {scale_factor:.2f} (height: {pil_img.height} -> {scaled_img.height})")
                else:
                    print(f"Image {idx} not scaled (height: {pil_img.height} <= {self.max_height})")
                    
            except Exception as e:
                print(f"Error converting image at index {idx} to PIL: {e}")
                continue

        if not valid_images:
            raise ValueError("No valid images found in the document.")
        print("======> original image size: ", valid_images[0].size)
        print("======> scaled image size: ", scaled_images[0].size)
        # Run detection on scaled images
        detec_prediction = self.detection_predictor(scaled_images)
        if len(detec_prediction) == 0:
            print("No detections found in the image with the size: ", scaled_images[0].size)

        all_langs = []
        all_slices = []
        all_polygons = []
        all_bboxes = []

        # Process detections and scale back to original size if needed
        for idx, (det_pred, image, scale_factor, lang) in enumerate(zip(detec_prediction, valid_images, scale_factors, ['en'])):
            # Scale polygons and bboxes back to original size if needed
            polygons = [
                self.pad_polygon(
                    self.scale_polygon(p.polygon, scale_factor),
                    image.width,
                    image.height
                ) for p in det_pred.bboxes
            ]
            bboxes = [
                self.pad_bbox(
                    self.scale_bbox(b.bbox, scale_factor),
                    image.width,
                    image.height
                ) for b in det_pred.bboxes
            ]
            
            # Process slices using original resolution image
            slices = self.slice_polys_from_image(image, polygons)
            
            all_langs.extend([lang] * len(slices))
            all_slices.extend(slices)
            all_polygons.extend(polygons)
            all_bboxes.extend(bboxes)
            
            # Create annotation on original size image
            annotated_img = np.array(image.copy())
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cv2.imwrite(f"annotated_image_{idx}.jpg", annotated_img)

        all_base64_images = [self.convert_to_base64(i) for i in all_slices]
        
        batch_size = 64
        ocr = []
        for i in range(0, len(all_base64_images), batch_size):
            batch_chunk = all_base64_images[i:i + batch_size]
            ocr.extend(self.get_recognition(batch_chunk, batch_size=batch_size))
        print("Ocrs:\n", ocr)
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
            'pages': [{
                'text_lines': result,
                'language': ['en'],
                'page': 1
            }],
        }
        
        with open("ocr_results.json", "w") as json_file:
            json.dump(d, json_file, indent=4)
        
        return d["pages"]
    
    def get_recognition_duplicate(self, image):
        return '0', 1, 1

    def get_recognition(self, images, batch_size=128):
        all_outputs = []
        for i in range(0, len(images), batch_size):
            chunk_images = images[i:i + batch_size]
            messages_batch = []
            for img in chunk_images:
                messages = [
                    {'role': 'system', 'content': "You are text extractor, you have extradinary ocr skills and can extract multiple languages, low opacity texts and cluttered text. Extact only the text from ths image, just respond with the extracted text alone. if you can't extract anything, respond with ''."},  # Your system message
                    {'role': 'user', 'content': [
                        {"type": "image", "image": f"data:image;base64,{img}"},
                        {"type": "text", "text": "Extract only the text from this image."},
                     ],
                    }     # Your user message
                ]
                messages_batch.append(messages)
            
            # Process the current chunk
            text_batch = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            image_inputs, video_inputs = process_vision_info(messages_batch)
            
            inputs = self.processor(
                text=text_batch,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                padding_side='left'
            ).to("cuda")
            
            # Generate with inference mode to save memory
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.processor.tokenizer.eos_token_id  # Ensure proper padding
                )
            
            # Decode outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            all_outputs.extend(output_texts)
            
            # Explicitly free memory
            del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
            torch.cuda.empty_cache()
        
        return all_outputs
    
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
    
    def __call__(self, base64: str, file_name: str):
        base64_bytes = self.decode_base64(base64)
        doc = DocImage(base64_bytes)
        output_filename = f'{file_name}-table.xlsx'
        print("Starting to process document...")
        doc.to_xlsx(
            dest=output_filename,
            ocr=self.ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=50
        )
        return output_filename