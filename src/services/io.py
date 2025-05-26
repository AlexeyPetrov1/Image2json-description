from typing import Tuple, Literal
from src.config.params import PROMPT_DIRECTORY, CLASSES_DIR
import os
from dataclasses import dataclass
from pathlib import Path


def get_default_classes(
    directory_with_classes: str = "user/default_humanoid_robot_classes.txt", sep="."
) -> Tuple[str, list[str]]:
    """Read default classes from input `.txt` file and set it as default.
    Classes in file will be separeted by `sep` argument"""
    with open(
        os.path.join(PROMPT_DIRECTORY, directory_with_classes),
        encoding="utf-8",
    ) as f:
        text_classes = f.read()
        classes = list(set(map(str.strip, text_classes.split(sep))))
        text_classes = ".".join(classes)
    return text_classes, classes


@dataclass
class ImageDataset:
    """Filepaths of all images in one folder"""

    folder_path: str | Path

    def get_images_list(
        self,
        available_formats: list[Literal[".png", ".jpg", ".jpeg"]] = (
            ".png",
            ".jpg",
            ".jpeg",
        ),
    ):
        return sorted(
            [
                Path(self.folder_path / f)
                for f in os.listdir(self.folder_path)
                if f.lower().endswith(available_formats)
            ]
        )

    def get_images_names_list(
        self,
        available_formats: list[Literal[".png", ".jpg", ".jpeg"]] = (
            ".png",
            ".jpg",
            ".jpeg",
        ),
    ):
        temp = sorted(
            [
                (Path(os.path.join(self.folder_path, f)), Path(f))
                for f in os.listdir(self.folder_path)
                if f.lower().endswith(available_formats)
            ]
        )
        return [f for _, f in temp]


@dataclass
class ImageInfo:
    path: str | Path
    classes: list[str]
    classes_str: str


class ImageInfoGetter:
    def __init__(self, image_dataset: ImageDataset, same_classes_on_all_images=False):
        self.image_dataset = image_dataset
        self.same_classes_on_all_images = same_classes_on_all_images

    def _get_info_of_one_image(self, image_path: str):
        try:
            path = image_path
            image_path = str(image_path)
            image_name = image_path[image_path.rfind("/") + 1 :]
            path = CLASSES_DIR / f"{image_name}.txt"
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                classes_list = list(set(text.split(" . ")))
                classes_str = ".".join(classes_list)
            return ImageInfo(path=path, classes=classes_list, classes_str=classes_str)
        except:
            return None

    def get_info_of_images(self):
        info_of_images = []
        image_files = self.image_dataset.get_images_list()
        default_classes_str, default_classes_list = get_default_classes()
        for image_path in image_files:
            if self.same_classes_on_all_images is False:
                saved_classes = self._get_info_of_one_image(image_path)
                info = (
                    saved_classes
                    if saved_classes is not None
                    else ImageInfo(
                        image_path, default_classes_list, default_classes_str
                    )
                )
            else:
                info = ImageInfo(image_path, default_classes_list, default_classes_str)
            info_of_images.append(info)
        return info_of_images
