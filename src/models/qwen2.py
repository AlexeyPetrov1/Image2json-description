from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.models.base import ModelInterface
from src.config.params import PROMPT_DIRECTORY, DEFAULT_DEVICE


class Qwen2Model(ModelInterface):
    """This class wraps interaction with Qwen2-VL MultiModal models, using HuggingFace API"""

    def __init__(
        self, model_id: str, torch_dtype="auto", device_map="auto", load_in_4bit=False
    ):
        self.model = None
        self.processor = None
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit

    def load(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            load_in_4bit=self.load_in_4bit,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self

    def _prepare_input(self, image_filepath, input_text, system_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_filepath,
                    },
                    {"type": "text", "text": input_text},
                ],
            }
        ]
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEFAULT_DEVICE)
        return inputs

    def predict(
        self,
        image_filename,
        input_text,
        system_prompt=None,
        max_new_tokens=1024,
        num_beams=4,
        temperature=1.0,
    ):
        inputs = self._prepare_input(image_filename, input_text, system_prompt)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
