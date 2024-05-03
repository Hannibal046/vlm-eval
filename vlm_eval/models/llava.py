"""
llava.py

Class definition for the LLaVa VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional likelihood
estimation.

Reference: https://github.com/haotian-liu/LLaVA/tree/main
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from accelerate import PartialState
import PIL
from PIL.Image import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vlm_eval.util.interfaces import VLM, ImageProcessor, Tokenizer

class LLaVa(VLM):
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
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        user_prompt = "USER: <image>\n{question} ASSISTANT:"
        self.template = system_prompt + " " + user_prompt
        self.model, self.processor = self.load()
        self.image_processor = None

        # Set Default Generation Configuration --> again from the Github Repository!
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.processor.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:

        """
        Loads model using a combination of `transformers.AutoModelForCausalLM` along with the special
        `LlavaLlamaForCausalLM` class defined in the `LLaVa` package.

        Using this instead of the default `LLaVa.load_pretrained_model` to remove bloat & patch image processor.

        Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/builder.py
        """
        with self.distributed_state.main_process_first():
            processor = AutoProcessor.from_pretrained(self.hub_path)
            model = LlavaForConditionalGeneration.from_pretrained(self.hub_path)

        # Load both the `model` and `vision_tower` onto the correct devices/in the correct precision!
        model = model.to(self.distributed_state.device).to(self.dtype)
        model.eval()

        return model, processor

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
        tally_qa_prompt_fn = self.get_mc_prompt_fn()
        ai2d_prompt_fn = self.get_mc_prompt_fn()

        return {
            "vqa-v2": vqa_prompt_fn,
            "gqa": vqa_prompt_fn,
            "vizwiz": vqa_uncertain_prompt_fn,
            "text-vqa": text_vqa_prompt_fn,
            "vsr": true_false_prompt_fn,
            "pope": vqa_prompt_fn,
            "tally-qa": tally_qa_prompt_fn,
            "refcoco": bbox_refer_prompt_fn,
            "ocid-ref": bbox_refer_prompt_fn,
            "ai2d": ai2d_prompt_fn,
            # Generic for GUI
            "captioning": captioning_prompt_fn,
            "bbox_pred": bbox_refer_prompt_fn,
            "vqa": vqa_prompt_fn,
            "true_false": true_false_prompt_fn,
        }[dataset_family]

    def get_captioning_prompt_fn(self) -> Callable[[str], str]:

        def llava_cap_prompt_fn() -> str:
            question = "Provide a short image description."
            return self.template.format_map(dict(question=question))

        return llava_cap_prompt_fn

    def get_vqa_chat_prompt_fn(self, uncertainty_aware: bool = False) -> Callable[[str], str]:
        """Generates the full reference prompt for VQA tasks."""

        q_prompt = ""
        if uncertainty_aware:
            q_prompt += "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
            q_prompt += "\nAnswer the question using a single word or phrase."

        # Otherwise, LLaVa-1.5 encourages short VQA responses by default.
        else:
            q_prompt += "\nAnswer the question using a single word or phrase."

        def llava_vqa_prompt_fn(question: str) -> str:
            question = question + q_prompt 
            return self.template.format_map(dict(question=question))

        return llava_vqa_prompt_fn

    def get_true_false_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a True/False captioning task."""

        # Construct True/False Prompt =>> Following InstructBLIP
        q_prompt_before = 'Based on the image, is this statement "True" or "False"?'
        q_prompt_after  = '\nRespond with "True" or "False" directly.'


        def llava_true_false_prompt_fn(caption: str) -> str:
            question = q_prompt_before + " " + caption + q_prompt_after
            return self.template.format_map(dict(question=question))

        return llava_true_false_prompt_fn

    def get_contrast_caption_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a multi-pair contrast captioning task (e.g., WinoGround)."""


        # Construct True/False Prompt =>> Following InstructBLIP
        cap_prompt = 'Does the following caption match the image? Caption: "{caption}"'
        cap_prompt += '\nRespond with "True" or "False" directly.'

        def llava_contrast_caption_prompt_fn(caption: str) -> str:
            question = cap_prompt.format_map(dict(caption=caption))
            return self.template.format_map(dict(question=question))

        return llava_contrast_caption_prompt_fn

    def get_mc_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a multiple-choice question-answer task."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!

        q_prompt = "{question}\n{choice_str}"
        q_prompt += "\nAnswer with the option's letter from the given choices directly."

        def llava_mc_prompt_fn(question: str, choices: List[str]) -> str:
            assert len(choices) <= 26, "Too many answer choices vs. possible letters in the alphabet!"
            choice_str = "\n".join([f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)])
            question = q_prompt.format_map(dict(question=question,choice_str=choice_str))
            return self.template.format_map(dict(question=question))

        return llava_mc_prompt_fn

    def get_bbox_refer_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a referring expression localization task."""

        # Construct Detection Prompt =>> Following LLaVa-1.5 Paper
        detect_prompt = 'Please provide the bounding box coordinate of the region this sentence describes: "{sentence}"'

        def llava_bbox_refer_prompt_fn(sentence: str) -> str:
            question = detect_prompt.format_map(dict(sentence=sentence))
            return self.template.format_map(dict(question=question))

        return llava_bbox_refer_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self, image_paths: List[str], questions: List[str], return_string_probabilities: Optional[List[str]] = None
    ) -> Union[List[str], List[List[float]]]:
        # By default, LLaVa code only neatly handles processing a single example at a time, due to the way the <image>
        # tokens are interleaved with the text; this code just loops over inputs (naive padding doesn't work...)
        gen_texts, gen_probabilities = [], []
        for image_path,question in zip(image_paths,questions):
            image = PIL.Image.open(image_path)
            inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.distributed_state.device)
        
            if return_string_probabilities is None:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    generate_ids = self.model.generate(**inputs, **self.generate_kwargs)
                    gen_text = self.processor.batch_decode(generate_ids[:,inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                gen_texts.append(gen_text)
            else:
                full_out_dict = self.model.generate(**inputs, **self.generate_kwargs,output_scores=True,return_dict_in_generate=True)

                # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                gen_ids = full_out_dict.sequences[0, inputs.input_ids.shape[1] :]

                # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                # Decode `gen_ids` and strip any <EOS> tokens
                gen_texts.append(self.processor.decode(gen_ids, skip_special_tokens=True).strip())

                # Get all token probabilities --> softmax over logits
                token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                # Get *normalized* probabilities for all values in `return_string_probabilities`
                slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                string_probs_unnormalized = token_probs[slice_idxs]
                string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities
