import json
import os
from src.config.params import (
    PROMPT_DIRECTORY,
    DETECTIONS_PICTURES_DIR,
    DETECTIONS_DIR,
    ANNOTATIONS_DIR,
    DATA_DIR,
    CLASSES_DIR,
)
from src.models.qwen2 import Qwen2Model
from tqdm import tqdm
from src.services.io import get_default_classes, ImageDataset


class DescriptionPipelines:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = Qwen2Model(self.model_id).load()

    def _run(self, image_filepath, text, system_prompt=None, max_new_tokens=1536):
        output_text = self.model.predict(
            image_filepath,
            text,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        return output_text[0]

    def get_final_json(self, image_name, save_output=True):
        with open(
            os.path.join(
                PROMPT_DIRECTORY, "system", "qwen_description_system_prompt.txt"
            ),
            encoding="utf-8",
        ) as f:
            system_prompt = f.read()
        print(image_name)
        with open(
            os.path.join(str(DETECTIONS_DIR), f"{image_name}.json"), encoding="utf-8"
        ) as file:
            json_info = json.load(file)["objects"]

        image_filepath = os.path.join(DETECTIONS_PICTURES_DIR, f"{image_name}.png")

        output_text = self._run(image_filepath, json_info, system_prompt=system_prompt)
        output_text = output_text.replace(
            "```json\n",
            "",
        ).replace("```", "")
        if save_output:
            try:
                text = json.loads(output_text)
                text["image_name"] = image_name
                with open(
                    ANNOTATIONS_DIR / f"{image_name}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(text, f, indent=5)
                    print("Saved! ", image_name)
            except Exception as ex:
                print(
                    "The file was not processed - it was not returned in json:",
                    image_name,
                    f"\n{output_text}",
                )
                print("EXCEPTION:\n", ex)

    # TODO: Make other pipelines to get classes, check, and so on.
    def get_image_classes(self, image_name, input_folder, save_output=True):
        with open(
            os.path.join(PROMPT_DIRECTORY, "system", "qwen_class_generator_prompt.txt"),
            encoding="utf-8",
        ) as f:
            system_prompt = f.read()
        text = "List all classes of objects on the picture:"
        image_filepath = os.path.join(DATA_DIR / input_folder, f"{image_name}")

        output_text = self._run(
            image_filepath, text, system_prompt=system_prompt, max_new_tokens=32
        )
        if save_output:
            try:
                print("IMAGE:", f"{image_name}", f"\n{CLASSES_DIR / image_name}.txt")
                with open(
                    CLASSES_DIR / f"{image_name}.txt", "w", encoding="utf-8"
                ) as f:
                    classes = list(set(output_text.split(" . ")))
                    classes = [
                        word
                        for word in classes
                        if 18 >= len(word) > 2
                        and "arm" not in word
                        and "robot" not in word
                        and "," not in word
                        and "." not in word
                    ]
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
                    text = " . ".join(classes)
                    f.write(text)
                    print("Saved! ", image_name)
            except Exception as ex:
                print(
                    "The file was not processed - it was not returned in json:",
                    image_name,
                )


if __name__ == "__main__":
    image_folder = ImageDataset(DATA_DIR / "debug")
    pipeline = DescriptionPipelines("Qwen/Qwen2-VL-2B-Instruct")
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(CLASSES_DIR, exist_ok=True)
    image_files = image_folder.get_images_names_list()
    for image_file in tqdm(image_files):
        pipeline.get_image_classes(
            image_file, image_folder.folder_path
        )  # OPTION 1: Find classes on images and save it
        # pipeline.get_final_json(
        #    image_file
        # )  # OPTION 2: Create description of detected_items (requires classes and detections of images)
