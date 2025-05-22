import os
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

from src.config.params import (
    MAX_IMAGES_PER_PREDICT,
    LANGSAM_BOX_THRESHOLD,
    DETECTIONS_DIR,
    DETECTIONS_PICTURES_DIR,
    CLASSES_DIR,
    DATA_DIR,
)
from src.models.langsam import LangSAMWrapper

from typing import List, Dict
from src.models.base import ModelInterface
from src.services.visualization import BoxFilter
from src.services.io import (
    get_default_classes,
    ImageDataset,
    ImageInfoGetter,
)


class ObjectDetector:
    def __init__(self, model: ModelInterface):
        self.model = model

    def load(self):
        self.model.load()
        return self

    def _save_one_file(
        self, prediction, filename, classes, image, box_filter_percent=0.0005
    ):
        scores, boxes, _, labels, _, _ = prediction.values()
        objects = []
        for label, score, box in list(zip(labels, scores, boxes)):
            if not BoxFilter(box_filter_percent).is_big_enough(box, image):
                continue
            if label not in classes:
                label = label.split(" ")
                label = " ".join(
                    label[round(len(label) / 2) :]
                )  # LangSAM sometimes returns combination of two classes. That fix it
            objects.append(
                {
                    "label": label,
                    "score": score.tolist(),
                    "box": list(box.astype(int).tolist()),
                }
            )
        one_file_result = {"filename": filename, "objects": objects}
        with open(
            os.path.join(DETECTIONS_DIR, f'{filename.split("/")[-1]}' + ".json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(one_file_result, f, indent=4, ensure_ascii=False)

    def detect(
        self,
        image_filenames: List,
        classes: List[str],
        box_threshold=LANGSAM_BOX_THRESHOLD,
        same_classes_on_all_images=True,
    ) -> List[Dict]:
        os.makedirs(DETECTIONS_DIR, exist_ok=True)
        classes = (
            classes
            if not same_classes_on_all_images
            else [classes[0] for _ in range(len(image_filenames))]
        )
        assert len(image_filenames) == len(
            classes
        ), "The number of images must correspond to the number of classes"
        interval = MAX_IMAGES_PER_PREDICT if same_classes_on_all_images is True else 1
        for i in range(0, len(image_filenames), interval):
            image_filenames_batch = image_filenames[
                i : i + interval
            ]  # TODO: Solve the problem with batching on LangSAM. The problem is when there are 2 images and 2 DIFFERENT classes
            classes_batch = classes[i : i + interval]
            images_batch = [
                Image.open(filename).convert("RGB")
                for filename in image_filenames_batch
            ]
            results = zip(
                image_filenames_batch,
                images_batch,
                classes_batch,
                self.model.predict(
                    images_batch, classes_batch, box_threshold=box_threshold
                ),
            )
            print("result_ok!", image_filenames_batch)
            for i, (filename, image, class_list, prediction) in enumerate(results):
                self._save_one_file(prediction, filename, classes, image)


class DetectionPipeline:
    def __init__(
        self,
        model,
        image_dataset: ImageDataset,
        same_classes_on_all_images=True,
        box_threshold=LANGSAM_BOX_THRESHOLD,
    ):
        self.model = model
        self.image_dataset = image_dataset
        self.same_classes_on_all_images = same_classes_on_all_images
        self.box_threshold = box_threshold

    def run(self, classes: None | list[str] = None) -> None:
        image_files = self.image_dataset.get_images_list()
        if classes is None:
            text_prompt, _ = get_default_classes()
            classes = [text_prompt]
        detector = ObjectDetector(self.model).load()
        detector.detect(
            image_files,
            classes,
            box_threshold=self.box_threshold,
            same_classes_on_all_images=self.same_classes_on_all_images,
        )

    def _mark_one_picture(self, filename, classes_of_picture):
        image = Image.open(filename).convert("RGB")
        fig, ax = plt.subplots()
        ax.imshow(image)
        result = json.load(
            open(os.path.join(DETECTIONS_DIR, f'{filename.split("/")[-1]}' + ".json"))
        )["objects"]
        for dictionary in result:
            label, _, box = dictionary.values()
            if label not in classes_of_picture:
                label = label.split(" ")
                label = " ".join(label[round(len(label) / 2) :])
            xmin, ymin, xmax, ymax = box
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    color="red",
                    linewidth=1,
                )
            )
            ax.text(
                xmin + (xmax - xmin) / 3,
                ymin,
                label,
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5),
            )
        marked_image_path = DETECTIONS_PICTURES_DIR / f'{filename.split("/")[-1]}.png'
        fig.savefig(marked_image_path)
        plt.close()

    def mark_pictures(
        self, classes: None | list[list[str]] = None
    ):  # TODO: Create a class/batch class object to interact without using indexes
        os.makedirs(DETECTIONS_PICTURES_DIR, exist_ok=True)
        image_files = self.image_dataset.get_images_list()
        if classes is None:
            _, classes = get_default_classes()
            classes = [classes for _ in range(len(image_files))]
        for i, filename in enumerate(image_files):
            self._mark_one_picture(filename, classes[i])


if __name__ == "__main__":
    model = LangSAMWrapper()
    folder = ImageDataset(DATA_DIR / "debug")
    same_classes_on_all_images = False
    pipeline = DetectionPipeline(
        model,
        image_dataset=folder,
        same_classes_on_all_images=same_classes_on_all_images,
    )
    classes_of_folder: list[str] = [
        info.classes_str
        for info in ImageInfoGetter(
            folder, same_classes_on_all_images=same_classes_on_all_images
        ).get_info_of_images()
    ]
    pipeline.run(classes=classes_of_folder)
    print(f"Детекция объектов завершена а и сохранена в папке {DETECTIONS_DIR}")
    pipeline.mark_pictures()
    print(f"Разметка картинок завершена и сохранена в папке {DETECTIONS_PICTURES_DIR}")
