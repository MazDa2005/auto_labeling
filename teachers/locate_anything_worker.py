import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Stateful worker with memory-safe image processing."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        dtype=torch.bfloat16,
        max_image_size: int = 1024,  # 🛡️ НОВОЕ: макс. размер изображения
        min_image_size: int = 448,   # 🛡️ НОВОЕ: мин. размер изображения
    ):
        self.device = device
        self.dtype = dtype
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 🛡️ КЛЮЧЕВОЕ: настраиваем процессор на ограничение размера изображения
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=True, 
            trust_remote_code=True,
            min_pixels=min_image_size * min_image_size,
            max_pixels=max_image_size * max_image_size,
        )
        
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="la_flash",
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 1024,  # 🛡️ УМЕНЬШЕНО с 2048 до 1024
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        # 🛡️ НОВОЕ: принудительный ресайз изображения ДО безопасного размера
        image = self._safe_resize(image)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        try:
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=verbose,
            )
        finally:
            # 🛡️ КРИТИЧЕСКИ ВАЖНО: освобождаем память после каждого вызова
            del inputs, pixel_values, input_ids, image_grid_hws
            torch.cuda.empty_cache()

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    def _safe_resize(self, image: Image.Image) -> Image.Image:
        """🛡️ Ограничивает размер изображения, сохраняя пропорции."""
        w, h = image.size
        max_dim = max(w, h)
        
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        elif max_dim < self.min_image_size:
            scale = self.min_image_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        return image

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    @staticmethod
    def parse_boxes_with_refs(answer: str, image_width: int, image_height: int) -> list[dict]:
        boxes = []
        answer = answer.replace('<|im_end|>', '').strip()
        pattern = r'<ref>(.*?)</ref>\s*<box>(.*?)</box>'

        for match in re.finditer(pattern, answer, re.DOTALL):
            class_name = match.group(1).strip()
            box_str = match.group(2).strip()

            if box_str.lower() == 'none' or not box_str:
                continue

            coords = re.findall(r'<(\d+)>', box_str)
            if len(coords) == 4:
                x1, y1, x2, y2 = [int(c) for c in coords]
                boxes.append({
                    "class": class_name,
                    "x1": x1 / 1000 * image_width,
                    "y1": y1 / 1000 * image_height,
                    "x2": x2 / 1000 * image_width,
                    "y2": y2 / 1000 * image_height,
                })

        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points
