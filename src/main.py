import os
from tqdm import tqdm
from src.config.params import (
    DETECTIONS_DIR,
    DETECTIONS_PICTURES_DIR,
    DATA_DIR,
    ANNOTATIONS_DIR,
    CLASSES_DIR,
)
from src.models.langsam import LangSAMWrapper
from src.core.detection import DetectionPipeline, ImageInfoGetter
from src.core.description import DescriptionPipelines
from src.services.io import ImageDataset


if __name__ == "__main__":

    PARAMETERS_COUNT = int(
        input("ENTER THE NUMBER OF PARAMETERS FOR QWEN2 (2, 7, 32, 72). One number: ")
    )
    if PARAMETERS_COUNT not in [2, 7, 32, 72]:
        print("INCORRECT NUMBER OF PARAMETERS ENTERED")
    else:
        same_classes_on_all_images = False
        model_name = f"Qwen/Qwen2-VL-{PARAMETERS_COUNT}B-Instruct"
        folder = input(
            "ENTER THE FOLDER NAME WITH THE IMAGES FOR ANALYSIS (by default, test_data_1 or debug in `data` folder):"
        )
        if "/" in folder:
            image_dataset = ImageDataset(folder)
        else:
            image_dataset = ImageDataset(DATA_DIR / folder)
        image_names = image_dataset.get_images_names_list()
        image_files = image_dataset.get_images_list()
        # Получение классов изображений
        pipeline = DescriptionPipelines(model_name)
        os.makedirs(CLASSES_DIR, exist_ok=True)
        for image_file in tqdm(image_names):
            pipeline.get_image_classes(image_file, image_dataset.folder_path)
        print(
            f"Getting the classes in the photo is completed! Classes saved in the {CLASSES_DIR} folder"
        )
        # Получение детекций и bounding_boxes изображений
        model = LangSAMWrapper()
        detection_pipe = DetectionPipeline(
            model,
            image_dataset=image_dataset,
            same_classes_on_all_images=same_classes_on_all_images,
        )
        # Читаем сохраненные классы на изображениях
        classes_of_folder: list[str] = [
            info.classes_str
            for info in ImageInfoGetter(
                image_dataset, same_classes_on_all_images=same_classes_on_all_images
            ).get_info_of_images()
        ]
        # Сохраняем в json детекции
        detection_pipe.run(classes=classes_of_folder)
        print(f"Object detection is complete! Saved in the {DETECTIONS_DIR} folder")
        # Наносим bboxs and labels на изображения для подачи в VLM
        detection_pipe.mark_pictures()
        print(
            f"The annotation of the images is complete! Saved in the {DETECTIONS_PICTURES_DIR} folder"
        )
        # Получаем финальные son
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
        for image_file in tqdm(image_names):
            pipeline.get_final_json(image_file)
