"""
instructblip.py

Class definition for the InstructBLIP VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional
likelihood estimation. Only supports the Vicuna LLM backbones (no FLAN-T5).
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import os
import time
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import AutoModelForCausalLM
from vlm_eval.util.interfaces import VLM, ImageProcessor, Tokenizer

## own
from .llava.conversation import conv_templates
from .llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from .llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info

# Define InstructBLIP Mapping from Model ID --> HF Hub Path
DEEPSEEKVL_MODELS = [
    "01-ai/Yi-VL-6B",
    "01-ai/Yi-VL-34B",
]
DEFAULT_IMAGE_TOKEN = "<image_placeholder>"

class YiVL(VLM):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        run_dir: Path,
        load_precision: str = "bf16",
        ocr: bool = False,
        max_length: int = 128,
        temperature: float = 0.2,
        **_: str,
    ) -> None:
        self.model_family, self.model_id, self.hub_path = model_family, model_id, run_dir
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[load_precision]
        self.ocr = ocr

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load Model on GPU(s) --> download if necessary via HF Hub
        self.model, self.tokenizer, image_processor = self.load()

        def process_image(image,**kwargs):
            if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
                image = expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
            return image_processor.preprocess(image,**kwargs)
            
        self.image_processor = process_image

        # For Fair Evaluation against LLaVa/Quartz/IDEFICS --> greedy decoding:
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # For computing likelihoods --> get tokens corresponding to "true", "false" and "yes", "no"
        self.string2idx = {}
        for trigger_string in ["true", "false", "yes", "no"]:
            token_idx_list = self.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Loads model and processors (InstructBLIPProcessor contains Vicuna Tokenizer, Q-Former Tokenizer, and an
        ImageProcessor) using the HF `InstructBLIP*.from_pretrained()` functionality.
        """
        with self.distributed_state.main_process_first():
            model_path = os.path.expanduser(self.hub_path)
            key_info["model_path"] = model_path
            tokenizer, model, image_processor, _ = load_pretrained_model(model_path)

        # Place Model on Device
        model = model.to(self.distributed_state.device, dtype=self.dtype)
        model.eval()

        return model, tokenizer, image_processor

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_fn(self, dataset_family: str = "vqa-v2") -> Callable[[str], str]:
        vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False)
        vqa_uncertain_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=True)
        true_false_prompt_fn = self.get_true_false_chat_prompt_fn()
        contrast_caption_prompt_fn = self.get_contrast_caption_chat_prompt_fn()
        bbox_refer_prompt_fn = self.get_bbox_refer_chat_prompt_fn()
        text_vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False, ocr_handling=True)
        captioning_prompt_fn = self.get_captioning_prompt_fn()

        return {
            "vqa-v2": vqa_prompt_fn,
            "gqa": vqa_prompt_fn,
            "vizwiz": vqa_uncertain_prompt_fn,
            "text-vqa": text_vqa_prompt_fn,
            "vsr": true_false_prompt_fn,
            "pope": vqa_prompt_fn,
            "refcoco": bbox_refer_prompt_fn,
            "ocid-ref": bbox_refer_prompt_fn,
            # Generic for GUI
            "captioning": captioning_prompt_fn,
            "bbox_pred": bbox_refer_prompt_fn,
            "vqa": vqa_prompt_fn,
            "true_false": true_false_prompt_fn,
        }[dataset_family]

    @staticmethod
    def get_captioning_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for captioning tasks."""

        def captioning_prompt_fn() -> str:
            return "A short image description:"

        return captioning_prompt_fn

    @staticmethod
    def get_vqa_chat_prompt_fn(uncertainty_aware: bool = False, ocr_handling: bool = False) -> Callable[[str], str]:
        """Generates the full reference prompt for VQA tasks."""
        
        def vqa_prompt_fn(question: str) -> str:
           question = DEFAULT_IMAGE_TOKEN + "\n" + question + "\nAnswer the question using a single word or phrase."
           conv_mode = 'mm_default'
           conv = conv_templates[conv_mode].copy()
           conv.append_message(conv.roles[0],question)
           conv.append_message(conv.roles[1], None)
           prompt = conv.get_prompt()
           return prompt

        return vqa_prompt_fn

    @staticmethod
    def get_true_false_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a True/False captioning task."""

        def true_false_prompt_fn(caption: str) -> str:
            return f'Based on the image, is this statement true or false? "{caption}" Answer:'

        return true_false_prompt_fn

    @staticmethod
    def get_contrast_caption_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a multi-pair contrast captioning task (e.g., WinoGround)."""

        def contrast_caption_prompt_fn(caption: str) -> str:
            return f'Does the following caption match the image (true or false)? Caption: "{caption}" Answer:'

        return contrast_caption_prompt_fn

    @staticmethod
    def get_bbox_refer_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a referring expression localization task."""

        def bbox_refer_prompt_fn(sentence: str) -> str:
            return f'Please provide the bounding box coordinate of the region this sentence describes: "{sentence}":'

        return bbox_refer_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self, pixel_values: torch.Tensor, questions: List[str], return_string_probabilities: Optional[List[str]] = None
    ) -> Union[List[str], List[List[float]]]:
        conv = conv_templates['mm_default'].copy()
        gen_texts = []
        # print(pixel_values.shape)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            for idx,question in enumerate(questions):
                input_ids = tokenizer_image_token(question,self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(pixel_values.device)
                image_tensor = pixel_values[idx]
                stop_str = conv.sep
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor[None,...],
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    **self.generate_kwargs,
                )
                input_token_len = input_ids.shape[1]
                outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()                
                gen_texts.append(outputs)
        return gen_texts
