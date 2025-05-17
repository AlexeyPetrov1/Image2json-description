
import json
import os
from src.config.params import PROMPT_DIRECTORY, DETECTIONS_PICTURES_DIR, DETECTIONS_DIR, ANNOTATIONS_DIR, DATA_DIR, CLASSES_DIR
from src.models.qwen2 import QwenModel
from tqdm import tqdm
from src.services.io import get_default_classes

class DescriptionPipelines:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = QwenModel(self.model_id).load()

    def _run(self, image_filepath, text, system_prompt=None, max_new_tokens=1536):
        output_text = self.model.predict(image_filepath, text, system_prompt=system_prompt, max_new_tokens=max_new_tokens)
        return output_text[0]

    def get_final_json(self, image_name, save_output=True):
        with open(os.path.join(PROMPT_DIRECTORY, 'system', "qwen_description_system_prompt.txt"), encoding="utf-8") as f:
            system_prompt = f.read()
        
        with open(os.path.join(str(DETECTIONS_DIR), f"{image_name}.json"), encoding="utf-8" ) as file:
            json_info = json.load(file)['objects']

        image_filepath = os.path.join(DETECTIONS_PICTURES_DIR, f'{image_name}.png')

        output_text = self._run(image_filepath, json_info, system_prompt=system_prompt)
        output_text = output_text.replace('```json\n', '', ).replace('```', '')
        if save_output:
            try:
                text = json.loads(output_text)
                text['image_name'] = image_name
                with open(ANNOTATIONS_DIR / f'{image_name}.json', "w", encoding="utf-8") as f:
                    json.dump(text, f, indent=5)
                    print("Сохранено! ", image_name)
            except:
                print("Файл не обработан - вернулся не json:", image_name, f'\n{output_text}')

    # TODO: Сделать другие пайплайны - для получения классов, проверки и тд
    def get_image_classes(self, image_name, save_output=True):
        with open(os.path.join(PROMPT_DIRECTORY, 'system', "qwen_class_generator_prompt.txt"), encoding="utf-8") as f:
            system_prompt = f.read()
        text = "List all classes of objects on the picture:"
        image_filepath = os.path.join(DATA_DIR / 'test_data_1', f'{image_name}')

        output_text = self._run(image_filepath, text, system_prompt=system_prompt, max_new_tokens=32)
        if save_output:
            try:
                with open(CLASSES_DIR / f'{image_name}.txt', "w", encoding="utf-8") as f:
                    classes = list(set(output_text.split(' . ')))
                    classes = [word for word in classes if 'arm' not in word and 'robot' not in word and 15>=len(word)>2 and ',' not in word and '.' not in word]
                    if len(classes) == 0:
                        _, def_classes = get_default_classes()
                        classes = def_classes
                    elif len(classes) == 1:
                        _, def_classes = get_default_classes()
                        def_classes.append(classes[0])
                        classes = def_classes
                    elif len(classes) == 2:
                        _, def_classes = get_default_classes()
                        def_classes.append(classes[0])
                        def_classes.append(classes[1])
                        classes = def_classes
                    text = ' . '.join(classes)
                    f.write(text)
                    print("Сохранено! ", image_name)
            except Exception as ex:
                print("Файл не обработан - вернулся не json:", image_name)


if __name__ == "__main__":
    image_folder = DATA_DIR / 'test_data_1'
    pipeline = DescriptionPipelines("Qwen/Qwen2-VL-7B-Instruct")
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(CLASSES_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 
    for image_file in tqdm(image_files):
        #pipeline.get_image_classes(image_file)
        pipeline.get_final_json(image_file)
