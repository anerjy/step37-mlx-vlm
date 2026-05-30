"""Image processor for Step-3.7-Flash with multi-crop ImagePatcher.

Step 3.7 input pipeline (matching upstream ``processing_step3.py``):

1. Pad to square if extreme aspect ratio with short edge < 32.
2. Resize so max(W, H) <= MAX_IMAGE_SIZE (3024).
3. Determine ``window_size`` (504 for >728 long edge, else 0 = single-view).
4. Crop into N square sub-patches (504×504 each), with ``patch_newline_mask``
   marking the last patch in each row.
5. Emit a single 728×728 ``global`` view + N×504² ``patch`` views.

Each image produces a variable number of pixel tensors:
  - global: 728×728 → 169 image tokens
  - patches: 504×504 × N → 81 image tokens each
  - newlines: inserted between rows

The chat_template.jinja translates the placeholder list into:
  <patch_start>[81×<im_patch>]<patch_end><patch_newline>?  ... <im_start>[169×<im_patch>]<im_end>

We register the processor by patching ``mlx_vlm.utils.load_processor`` on
import: when ``config.json``'s top-level ``model_type`` is ``step3p7`` and
``transformers.AutoProcessor.from_pretrained`` can't resolve a class for
it, we build a ``Step3p7Processor`` here using the model's stock fast
tokenizer.
"""
from __future__ import annotations

import logging
from itertools import product
from math import ceil
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_flat_list_of_images
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)


# OpenAI CLIP normalisation. Upstream stepfun-ai processing_step3.py uses
# these exact constants; ImageNet stats differ enough to mis-color the
# vision tower output.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGE_SIZE = 728
PATCH_SIZE = 504
MAX_IMAGE_SIZE = 3024

NUM_IMAGE_FEATURE_SIZE = 169   # 728 / 14 / 2 / 2 → 13 → 13²
NUM_PATCH_FEATURE_SIZE = 81    # 504 / 14 / 2 / 2 → 9  → 9²


def _normalize_pil(image: Image.Image, size: int) -> np.ndarray:
    """PIL → (3, size, size) float32 CHW, OpenAI CLIP normalized.

    Matches upstream Step3VisionProcessor at processing_step3.py:60 with
    BILINEAR interpolation (we don't depend on torchvision/bicubic here —
    upstream defaults to bilinear when constructed with ``"bilinear"``).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return arr


class _ImagePatcher:
    """Pure-PIL/numpy port of upstream ImagePatcher (processing_step3.py:93).

    Outputs (global_img, [patch1, patch2, ...], [newline_mask_per_patch]).
    """

    @staticmethod
    def determine_window_size(long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, PATCH_SIZE) if long / short > 4 else PATCH_SIZE

    @staticmethod
    def square_pad(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    @staticmethod
    def get_image_size_for_padding(img_width: int, img_height: int) -> Tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    @staticmethod
    def get_image_size_for_preprocess(img_width: int, img_height: int) -> Tuple[int, int]:
        if max(img_height, img_width) > MAX_IMAGE_SIZE:
            scale_factor = MAX_IMAGE_SIZE / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
        return img_width, img_height

    @staticmethod
    def get_image_size_for_crop(img_width: int, img_height: int, window_size: int) -> Tuple[int, int]:
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size
        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    @staticmethod
    def slide_window(width: int, height: int, sizes: List[Tuple[int, int]],
                     steps: List[Tuple[int, int]]
                     ) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
        windows = []
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step
            x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w
            y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h
            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)
        return [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])) for b in windows], (x_num, y_num)

    @staticmethod
    def patch_crop(img: Image.Image, i: int, j: int, th: int, tw: int) -> Image.Image:
        return img.crop((j, i, j + tw, i + th))

    def __call__(self, img: Image.Image
                 ) -> Tuple[Image.Image, List[Image.Image], Optional[List[bool]]]:
        img_width, img_height = img.size
        new_img_width, new_img_height = self.get_image_size_for_padding(img_width, img_height)
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_img_width, new_img_height = self.get_image_size_for_preprocess(img_width, img_height)
        img = img.resize((new_img_width, new_img_height), Image.BILINEAR)
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width),
            min(new_img_height, new_img_width),
        )
        if window_size == 0:
            return img, [], None

        new_img_width, new_img_height = self.get_image_size_for_crop(
            new_img_width, new_img_height, window_size,
        )
        if (new_img_width, new_img_height) != img.size:
            img_for_crop = img.resize((new_img_width, new_img_height), Image.BILINEAR)
        else:
            img_for_crop = img

        patches: List[Image.Image] = []
        newlines: List[int] = []
        center_list, (x_num, _y_num) = self.slide_window(
            new_img_width, new_img_height,
            [(window_size, window_size)], [(window_size, window_size)],
        )
        for patch_id, center_lf_point in enumerate(center_list):
            x, y, patch_w, patch_h = center_lf_point
            big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)
            patches.append(big_patch)
            if (patch_id + 1) % x_num == 0:
                newlines.append(patch_id)
        # Last newline marker is redundant (image ends, no more rows below).
        if newlines and newlines[-1] == len(patches) - 1:
            newlines.pop()
        return img, patches, [i in newlines for i in range(len(patches))] if len(patches) > 0 else None


class Step3p7ImageProcessor(BaseImageProcessor):
    """Multi-crop image processor. Outputs (global, patches, num_patches, newline mask)."""

    model_input_names = [
        "pixel_values",            # (B_global, 3, IMAGE_SIZE, IMAGE_SIZE) — one row per image, global view
        "patch_pixel_values",      # (B_patch, 3, PATCH_SIZE, PATCH_SIZE) — concatenated across images
        "num_patches",             # list[int] length = num_images
        "patch_newline_mask",      # 1-D bool array length = sum(num_patches)
    ]

    def __init__(self, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patcher = _ImagePatcher()

    def preprocess(self, images: ImageInput, return_tensors: Optional[str] = None, **kwargs) -> BatchFeature:
        images = make_flat_list_of_images(images)
        global_arrs: List[np.ndarray] = []
        patch_arrs: List[np.ndarray] = []
        num_patches: List[int] = []
        newline_mask: List[bool] = []
        for img in images:
            global_img, patches, mask = self.patcher(img)
            global_arrs.append(_normalize_pil(global_img, size=self.image_size))
            num_patches.append(len(patches))
            for p in patches:
                patch_arrs.append(_normalize_pil(p, size=self.patch_size))
            if mask is not None:
                newline_mask.extend(mask)
        data = {
            "pixel_values": np.stack(global_arrs, axis=0),
            "num_patches": np.array(num_patches, dtype=np.int32),
        }
        if patch_arrs:
            data["patch_pixel_values"] = np.stack(patch_arrs, axis=0)
        if newline_mask:
            data["patch_newline_mask"] = np.array(newline_mask, dtype=bool)
        return BatchFeature(data=data, tensor_type=return_tensors)


class Step3p7Processor(ProcessorMixin):
    """Tokenizer + image_processor bundle. ProcessorMixin compatible.

    Handles per-image placeholder expansion:
       [<im_patch>] →  <patch_start>[81×<im_patch>]<patch_end>[<patch_newline>]?
                       ×N
                       + <im_start>[169×<im_patch>]<im_end>
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = self.image_token * NUM_IMAGE_FEATURE_SIZE
        self.patch_feature_placeholder = self.image_token * NUM_PATCH_FEATURE_SIZE

    def _per_image_repl(self, num_patches: int,
                        patch_newline_mask_for_image: Optional[List[bool]]) -> str:
        text = ""
        if num_patches > 0:
            for i in range(num_patches):
                text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
                if patch_newline_mask_for_image and patch_newline_mask_for_image[i]:
                    text += "<patch_newline>"
        text += f"<im_start>{self.image_feature_placeholder}<im_end>"
        return text

    def _replace_image_token_in_text(self, text: str, per_image_repls: List[str]) -> str:
        # Each "<im_patch>" in the user text expands into the per-image repl.
        # The chat_template emits one "<im_patch>" per image during the user
        # turn, so split-and-rejoin works.
        parts = text.split(self.image_token)
        if len(parts) - 1 != len(per_image_repls):
            raise ValueError(
                f"Number of image placeholders ({len(parts) - 1}) != number of images ({len(per_image_repls)})"
            )
        out = [parts[0]]
        for i, repl in enumerate(per_image_repls):
            out.append(repl)
            out.append(parts[i + 1])
        return "".join(out)

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[ImageInput] = None,
        return_tensors: Optional[str] = "np",
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Must provide at least one of text or images")

        out: dict = {}

        if images is not None:
            img_features = self.image_processor.preprocess(images, return_tensors=None)
            for k, v in img_features.data.items():
                out[k] = v
            num_patches_list = list(img_features.data.get("num_patches", []))
            patch_newline_mask = img_features.data.get("patch_newline_mask")
            # Slice per-image newline mask
            per_image_masks: List[Optional[List[bool]]] = []
            offset = 0
            for n in num_patches_list:
                if n == 0:
                    per_image_masks.append(None)
                elif patch_newline_mask is not None:
                    per_image_masks.append(list(patch_newline_mask[offset:offset + n]))
                    offset += n
                else:
                    per_image_masks.append([False] * n)
            per_image_repls = [
                self._per_image_repl(num_patches_list[i], per_image_masks[i])
                for i in range(len(num_patches_list))
            ]

            if text is not None:
                if isinstance(text, str):
                    text = [text]
                # Per-prompt: expand <im_patch> markers using per_image_repls
                # in left-to-right order. Single prompt = all images go to it.
                if len(text) == 1:
                    text[0] = self._replace_image_token_in_text(text[0], per_image_repls)
                else:
                    # multi-prompt batch: 1 image per prompt position assumed
                    text = [
                        self._replace_image_token_in_text(t, [per_image_repls[i]])
                        for i, t in enumerate(text)
                    ]

        if text is not None:
            tok = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            tok_data = tok.data if hasattr(tok, "data") else tok
            for k, v in tok_data.items():
                out[k] = v

        return BatchFeature(data=out, tensor_type=return_tensors)


# --- side-effect patch to mlx_vlm.utils.load_processor --------------------

def _is_step3p7_model_dir(model_path) -> bool:
    import json
    from pathlib import Path
    p = Path(model_path) / "config.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("model_type") == "step3p7"
    except Exception:
        return False


def _build_step3p7_processor(model_path) -> "Step3p7Processor":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    image_processor = Step3p7ImageProcessor(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    proc = Step3p7Processor(image_processor=image_processor, tokenizer=tokenizer)
    from mlx_vlm.models.base import load_chat_template
    load_chat_template(tokenizer, model_path)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        proc.chat_template = tokenizer.chat_template
    return proc


_PATCHED = False


def apply_processor_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    import mlx_vlm.utils as _mu
    orig = _mu.load_processor

    def patched_load_processor(model_path, add_detokenizer=True, eos_token_ids=None, **kwargs):
        if _is_step3p7_model_dir(model_path):
            proc = _build_step3p7_processor(model_path)
            if add_detokenizer:
                from mlx_vlm.utils import load_tokenizer
                detok_cls = load_tokenizer(model_path, return_tokenizer=False)
                tok = proc.tokenizer
                proc.detokenizer = detok_cls(tok)
                from mlx_vlm.utils import StoppingCriteria
                final_eos = (
                    eos_token_ids if eos_token_ids is not None
                    else getattr(tok, "eos_token_ids", None)
                )
                criteria = StoppingCriteria(final_eos, tok)
                tok.stopping_criteria = criteria
            return proc
        return orig(model_path, add_detokenizer=add_detokenizer, eos_token_ids=eos_token_ids, **kwargs)

    _mu.load_processor = patched_load_processor
    _PATCHED = True
    logger.info("mlx_vlm.utils.load_processor patched for step3p7 multi-crop")
    return True
