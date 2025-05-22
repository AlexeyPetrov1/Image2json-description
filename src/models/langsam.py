from src.models.base import ModelInterface, NotLoadedModelError
from lang_sam import LangSAM
from src.config.params import LANGSAM_DEFAULT_MODEL_SAM, LANGSAM_BOX_THRESHOLD


class LangSAMWrapper(ModelInterface):
    """Language Segment-Anything is an open-source project that combines the power
    of instance segmentation and text prompts to generate masks for specific objects in images.
    Built on the recently released Meta model, Segment Anything Model 2,
    and the GroundingDINO detection model, it's an easy-to-use and effective tool
    for object detection and image segmentation.
    Official documentation: https://github.com/luca-medeiros/lang-segment-anything
    """

    def __init__(self, sam_type: str = LANGSAM_DEFAULT_MODEL_SAM):
        self.sam_type = sam_type
        self.model = None

    def load(self):
        self.model = LangSAM(sam_type=self.sam_type)
        return self

    def predict(
        self,
        images: list,
        classes: list[str] | list[str],
        box_threshold: float = LANGSAM_BOX_THRESHOLD,
    ):
        return self.model.predict(images, classes, box_threshold=box_threshold)
