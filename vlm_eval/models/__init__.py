from pathlib import Path
from typing import Optional

from vlm_eval.util.interfaces import VLM

from .instructblip import InstructBLIP
# from .llava import LLaVa
from .prismatic import PrismaticVLM
from .deepseek_vl import DeepSeekVL
from .yi_vl import YiVL
from .intern_vl import InternVL

# === Initializer Dispatch by Family ===
FAMILY2INITIALIZER = {
    "instruct-blip": InstructBLIP, 
    # "llava-v15": LLaVa, 
    "prismatic": PrismaticVLM, 
    "deepseek_vl":DeepSeekVL,
    "yi_vl":YiVL,
    "intern_vl":InternVL,
    }


def load_vlm(
    model_family: str,
    model_id: str,
    run_dir: Path,
    hf_token: Optional[str] = None,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length=128,
    temperature=1.0,
) -> VLM:
    assert model_family in FAMILY2INITIALIZER, f"Model family `{model_family}` not supported!"
    return FAMILY2INITIALIZER[model_family](
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
    )
