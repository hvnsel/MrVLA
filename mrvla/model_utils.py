"""Load OpenVLA, locate its decoder layers, and run inference while capturing activations."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# OpenVLA's fixed prompt template (see openvla/openvla experiments).
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def load_openvla(model_name: str, device: str = "cuda:0", use_flash_attn: bool = True):
    """Return (model, processor) for an OpenVLA checkpoint in bf16."""
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, processor


def locate_decoder_layers(model) -> torch.nn.ModuleList:
    """Find the LLM decoder-layer stack inside an OpenVLA model.

    Tries the known OpenVLA/Llama attribute paths first, then falls back to scanning
    for a ModuleList whose entries look like transformer decoder layers.
    """
    candidate_paths = [
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in candidate_paths:
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj

    # Fallback: scan modules for a ModuleList of *DecoderLayer blocks.
    best = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            first = module[0]
            if "DecoderLayer" in type(first).__name__:
                if best is None or len(module) > len(best):
                    best = module
    if best is not None:
        return best

    raise RuntimeError(
        "Could not locate the decoder layer stack. Inspect model.named_modules() "
        "and pass an explicit path."
    )


def get_hidden_dim(layers: torch.nn.ModuleList) -> int:
    """Infer hidden size from the first decoder layer's parameters."""
    for p in layers[0].parameters():
        if p.dim() >= 1:
            return p.shape[-1] # [B, S, H] -> return H, the hidden dimension
    raise RuntimeError("Could not infer hidden dimension from decoder layer.")


def build_inputs(processor, image: Image.Image, instruction: str, device: str):
    """Tokenize the OpenVLA prompt + image into model inputs (bf16)."""
    prompt = PROMPT_TEMPLATE.format(instruction=instruction.lower().strip().rstrip("."))
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    return inputs


@torch.no_grad()
def predict_and_capture(
    model,
    processor,
    collector, # expects ActivationCollector instance
    image: Image.Image,
    instruction: str,
    device: str,
    unnorm_key: str | None = None,
    action_dim: int = 7,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Run one inference step, capturing pooled activations for every hooked layer.

    Returns (action, acts) where:
      - action: np.ndarray [action_dim] if unnorm_key is set, else None
      - acts:   np.ndarray [num_layers, hidden] (float depends on collector dtype)
    """
    inputs = build_inputs(processor, image, instruction, device)
    collector.reset()

    # OpenVLA's predict_action() appends a trailing token (29871) to input_ids when the
    # prompt does not already end with it, but it does NOT extend any attention_mask we
    # pass in. That leaves the attention mask one token shorter than the multimodal
    # embeddings (text + 256 image patches), which crashes the Llama attention with an
    # off-by-one (e.g. 279 vs 278). Drop attention_mask so generate() rebuilds a
    # correctly-sized one AFTER the append.
    inputs.pop("attention_mask", None)

    action = None
    if unnorm_key is not None:
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        action = np.asarray(action, dtype=np.float32)
    else:
        # No normalization stats requested: drive a forward pass to trigger the hooks.
        model.generate(**inputs, max_new_tokens=action_dim, do_sample=False)

    acts = collector.gather_single()  # [L, H]
    return action, acts
