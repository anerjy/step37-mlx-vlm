from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .step3p7 import Model
from .vision import StepRoboticsVisionEncoder

# Patch mlx_vlm.utils.load_processor so our model dirs return a working
# Step3p7Processor instead of bombing on AutoProcessor (transformers
# doesn't know about step3p7).
try:
    from .processing_step3p7 import apply_processor_patch as _apply_proc_patch
    _apply_proc_patch()
except Exception:
    import logging
    logging.getLogger(__name__).exception("step3p7 processor patch failed")
