import os
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

from src.config.params import PROMPT_DIRECTORY, MAX_IMAGES_PER_PREDICT, LANGSAM_BOX_THRESHOLD, DETECTIONS_DIR, DETECTIONS_PICTURES_DIR, CLASSES_DIR
from src.models.langsam import LangSAMWrapper

from typing import List, Dict, Tuple
from src.models.base import ModelInterface
from src.services.visualization import BoxFilter
from src.services.io import get_default_classes



class ObjectDetector:
    def __init__(self, model: ModelInterface):
        self.model = model

    def load(self):
        self.model.load()
        return self


    def detect(self, image_filenames: List, classes: List[str], box_threshold=LANGSAM_BOX_THRESHOLD, same_classes_on_all_images=True) -> List[Dict]:
        os.makedirs(DETECTIONS_DIR, exist_ok=True)
        classes = [classes[0] for _ in range(len(image_filenames))] if same_classes_on_all_images else classes
        assert len(image_filenames) == len(classes), "Число изображений должно соответствовать числу классов"
        interval = MAX_IMAGES_PER_PREDICT if same_classes_on_all_images is True else 1
        for i in range(0, len(image_filenames)):
            image_filenames_batch = image_filenames[i:i+interval] # TODO: разобраться с батчеванием. Проблема, когда 2 изображения и 2 РАЗНЫХ класса
            print(image_filenames_batch)
            classes_batch = classes[i:i+interval]
            images_batch = [Image.open(filename).convert("RGB") for filename in image_filenames_batch]
            results = zip(
                image_filenames_batch, classes_batch, 
                self.model.predict(images_batch, classes_batch, box_threshold=box_threshold)
            )
            print("result_ok!", image_filenames_batch)
            for i, (filename, class_list, result) in enumerate(results):
                scores, boxes, _, labels, _, _ = result.values()
                objects = []
                for label, score, box in list(zip(labels, scores, boxes)):
                    if not BoxFilter(0.0005).is_big_enough(box, images_batch[i]):
                        continue
                    if label not in classes:
                        label = label.split(' ')
                        label = ' '.join(label[round(len(label)/2):])
                    objects.append({"label":label, "score": score.tolist(), "box": list(box.astype(int).tolist())})
                one_file_result = {"filename" : filename, "objects": objects}
                with open(os.path.join(DETECTIONS_DIR, f'{filename.split("/")[-1]}' + '.json'), "w", encoding="utf-8") as f:
                    json.dump(one_file_result, f, indent=4, ensure_ascii=False)   


class DetectionPipeline:
    def __init__(self, model, image_folder:str= "data/test_data_1", same_classes_on_all_images=True, box_threshold=LANGSAM_BOX_THRESHOLD):
        self.model = model
        self.image_folder = image_folder
        self.same_classes_on_all_images=same_classes_on_all_images
        self.box_threshold = box_threshold


    def get_classes_of_folder(self):
        classes_of_folder = []
        image_files = sorted([f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        default_classes, _ = get_default_classes()
        for image_name in image_files:
            try:
                with open(CLASSES_DIR / f'{image_name}.txt', "r", encoding="utf-8") as f:
                    text =f.read()
                    classes = '.'.join(list(set(text.split(' . '))))
                    #classes_of_folder.update({f"{image_name}" : classes})
                    classes_of_folder.append(classes)
            except Exception as ex:
                #classes_of_folder.update({f"{image_name}" : default_classes})
                classes_of_folder.append(default_classes)
        return classes_of_folder


    def detection_pipeline(self, classes: None | list[str]=None) -> None:
        image_files = sorted([os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if classes is None:
            text_prompt, _ = get_default_classes()
            classes = [text_prompt]
        detector = ObjectDetector(self.model).load()
        detector.detect(image_files, classes, box_threshold=self.box_threshold, same_classes_on_all_images=self.same_classes_on_all_images)


    def mark_pictures(self, classes: None | list[list[str]]=None): # TODO: Создать объект класса / батча класса, чтобы взаимодействовать не через индексы
        os.makedirs(DETECTIONS_PICTURES_DIR, exist_ok=True)
        image_files = sorted([os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if classes is None:
            _, classes = get_default_classes()
            classes = [classes for _ in range(len(image_files))]
        for i, filename in enumerate(image_files):
            image = Image.open(filename).convert("RGB")
            fig, ax = plt.subplots()
            ax.imshow(image)
            result = json.load(open(os.path.join(DETECTIONS_DIR, f'{filename.split("/")[-1]}' + '.json')))["objects"]
            for dictionary in result:
                label, _,box = dictionary.values()
                if label not in classes[i]:
                    label = label.split(' ')
                    label = ' '.join(label[round(len(label)/2):])
                xmin, ymin, xmax, ymax = box
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=1))
                ax.text(xmin+(xmax-xmin)/3, ymin, label, color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            marked_image_path = DETECTIONS_PICTURES_DIR / f'{filename.split("/")[-1]}.png'
            fig.savefig(marked_image_path)
            plt.close()
        

if __name__ == "__main__":
    model = LangSAMWrapper()
    folder = "data/test_data_1"
    pipeline = DetectionPipeline(model, image_folder=folder, same_classes_on_all_images=False)
    classes_of_folder = pipeline.get_classes_of_folder()
    pipeline.detection_pipeline(classes=classes_of_folder)
    print(f"Детекция объектов завершена а и сохранена в папке {DETECTIONS_DIR}")
    pipeline.mark_pictures()
    print(f"Разметка картинок завершена и сохранена в папке {DETECTIONS_PICTURES_DIR}")
