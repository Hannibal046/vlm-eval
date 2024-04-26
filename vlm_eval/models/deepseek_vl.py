"""
instructblip.py

Class definition for the InstructBLIP VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional
likelihood estimation. Only supports the Vicuna LLM backbones (no FLAN-T5).
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from accelerate import PartialState
from vlm_eval.util.interfaces import VLM, ImageProcessor, Tokenizer

## Model-specific Import
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.models.processing_vlm import VLChatProcessorOutput
from deepseek_vl.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

DEEPSEEKVL_MODELS = [
    "deepseek-ai/deepseek-vl-7b-chat",
    "deepseek-ai/deepseek-vl-1.3b-chat",
]
DEFAULT_IMAGE_TOKEN = "<image_placeholder>"

class DeepSeekVL(VLM):
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
        self.model, self.text_img_processor, image_processor = self.load()

        ## deepseek-vl image_processor need List[Image] 
        self.image_processor = lambda image,**kwargs:image_processor([image],**kwargs)

        # For Fair Evaluation against LLaVa/Quartz/IDEFICS --> greedy decoding:
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # For computing likelihoods --> get tokens corresponding to "true", "false" and "yes", "no"
        self.string2idx = {}
        for trigger_string in ["true", "false", "yes", "no"]:
            token_idx_list = self.text_img_processor.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Loads model and processors (InstructBLIPProcessor contains Vicuna Tokenizer, Q-Former Tokenizer, and an
        ImageProcessor) using the HF `InstructBLIP*.from_pretrained()` functionality.
        """
        with self.distributed_state.main_process_first():
            text_img_processor = VLChatProcessor.from_pretrained(self.hub_path)
            model = AutoModelForCausalLM.from_pretrained(self.hub_path)

        # Lift `image_processor` for use in evaluation harnesses
        image_processor = text_img_processor.image_processor

        # Place Model on Device
        model = model.to(self.distributed_state.device, dtype=self.dtype)
        model.eval()

        return model, text_img_processor, image_processor

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_fn(self, dataset_family: str = "vqa-v2") -> Callable[[str], str]:
        vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False)
        vqa_uncertain_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=True)
        true_false_prompt_fn = self.get_true_false_chat_prompt_fn()
        contrast_caption_prompt_fn = self.get_contrast_caption_chat_prompt_fn()
        bbox_refer_prompt_fn = self.get_bbox_refer_chat_prompt_fn()
        text_vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False)
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
        """
        Generates the full reference prompt for captioning tasks.
        Same as LLaVA
        """

        def captioning_prompt_fn() -> str:
            return "A short image description:"

        return captioning_prompt_fn

    @staticmethod
    def get_vqa_chat_prompt_fn(uncertainty_aware: bool = False) -> Callable[[str], str]:
        """Generates the full reference prompt for VQA tasks."""

        def vqa_prompt_fn(question: str) -> str:
           prompt =  f"{DEFAULT_IMAGE_TOKEN}{question}"
           if uncertainty_aware:
               prompt += "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
           prompt += "\nAnswer the question using a single word or phrase."
           return prompt

        return vqa_prompt_fn

    @staticmethod
    def get_true_false_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a True/False captioning task."""

        def true_false_prompt_fn(caption: str) -> str:
            cap_prompt = f'{DEFAULT_IMAGE_TOKEN}\nBased on the image, is this statement "True" or "False"? {caption}'
            cap_prompt += '\nRespond with "True" or "False" directly.'
            return cap_prompt

        return true_false_prompt_fn

    @staticmethod
    def get_contrast_caption_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a multi-pair contrast captioning task (e.g., WinoGround)."""

        def contrast_caption_prompt_fn(caption: str) -> str:
            return f'{DEFAULT_IMAGE_TOKEN}Does the following caption match the image (true or false)? Caption: "{caption}" Answer:'

        return contrast_caption_prompt_fn

    @staticmethod
    def get_bbox_refer_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a referring expression localization task."""

        def bbox_refer_prompt_fn(sentence: str) -> str:
            return f'{DEFAULT_IMAGE_TOKEN}Please provide the bounding box coordinate of the region this sentence describes: "{sentence}":'

        return bbox_refer_prompt_fn

    def deepseek_vl_prepare(self,question:str,image: Path):
        ## single batch inference
        conversation = [
            {"role": "User","content": question,},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self.text_img_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.text_img_processor.sft_format,
            system_prompt=self.text_img_processor.system_prompt,
        )
        input_ids = self.text_img_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)
        image_token_mask: torch.BoolTensor = input_ids == self.text_img_processor.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.text_img_processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )
        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            pixel_values=pixel_values[None,...],
            num_image_tokens=num_image_tokens,
        )
        prepare = self.text_img_processor.batchify([prepare])
        return prepare

    @torch.inference_mode()
    def generate_answer(
        self, image_path: List[Path], questions: List[str], return_string_probabilities: Optional[List[str]] = None
    ) -> Union[List[str], List[List[float]]]:
        gen_texts = []
        with torch.cuda.amp.autocast(dtype=self.dtype):
            for image,question in zip(image_path,questions):
                conversation = [
                    {
                        "role": "User",
                        "content": question,
                        "images": [image],
                    },
                    {"role": "Assistant", "content": ""},
                ]
                pil_images = load_pil_images(conversation)
                prepare_inputs = self.text_img_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True
                ).to(self.distributed_state.device)
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.text_img_processor.tokenizer.eos_token_id,
                    bos_token_id=self.text_img_processor.tokenizer.bos_token_id,
                    eos_token_id=self.text_img_processor.tokenizer.eos_token_id,
                    use_cache=True,
                    **self.generate_kwargs,
                )
                answer = self.text_img_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                gen_texts.append(answer)
        return gen_texts
