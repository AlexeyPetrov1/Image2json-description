import os
from tqdm import tqdm
from src.config.params import DETECTIONS_DIR, DETECTIONS_PICTURES_DIR, DATA_DIR, ANNOTATIONS_DIR, CLASSES_DIR
from src.models.langsam import LangSAMWrapper
from src.core.detection import DetectionPipeline
from src.core.description import DescriptionPipelines


if __name__=="__main__":
    # Получение классов изображений
    PARAMETERS_COUNT = int(input("ВВЕДИТЕ ЧИСЛО ПАРАМЕТРОВ ДЛЯ QWEN2 (2, 7, 32, 72). Одно число:"))
    if PARAMETERS_COUNT not in [2, 7, 32, 72]  :
        print("ВВЕДЕНО НЕВЕРНОЕ ЧИСЛО ПАРАМТЕРОВ")
    else:
        model_name = f"Qwen/Qwen2-VL-{PARAMETERS_COUNT}B-Instruct"
        folder = input("ВВЕДИТЕ ПАПКУ С ИЗОБРАЖЕНИЯМИ ДЛЯ АНАЛИЗА (по умолчанию test_data_1 или debug): ")
        image_folder = DATA_DIR / folder
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        pipeline = DescriptionPipelines(model_name)
        os.makedirs(CLASSES_DIR, exist_ok=True)
        for image_file in tqdm(image_files):
            pipeline.get_image_classes(image_file)
        print(f"Получение классов на фото завершено и сохранена в папке {CLASSES_DIR}")
        # Получение детекций и bounding_boxes изображений
        model = LangSAMWrapper()
        detection_pipe = DetectionPipeline(model, image_folder=image_folder, same_classes_on_all_images=False)
        # Читаем сохраненные классы на изображениях
        classes_of_folder = detection_pipe.get_classes_of_folder()
        # Сохраняем в json детекции
        detection_pipe.detection_pipeline(classes=classes_of_folder)
        print(f"Детекция объектов завершена и сохранена в папке {DETECTIONS_DIR}")
        # Наносим bboxs and labels на изображения для подачи в VLM
        detection_pipe.mark_pictures()
        print(f"Разметка картинок завершена и сохранена в папке {DETECTIONS_PICTURES_DIR}")
        # Получаем финальные son
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
        for image_file in tqdm(image_files):
            pipeline.get_final_json(image_file)
