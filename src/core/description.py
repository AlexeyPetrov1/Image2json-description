import json
import os
import yaml
from PIL import Image
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
from src.config.params import (
    PROMPT_DIRECTORY,
    DETECTIONS_PICTURES_DIR,
    DETECTIONS_DIR,
    ANNOTATIONS_DIR,
    DATA_DIR,
    CLASSES_DIR,
)
from abc import abstractmethod, ABC
from src.models.qwen2 import Qwen2Model
from tqdm import tqdm
from src.services.io import get_default_classes, ImageDataset


class MultiModalPipeline(ABC):
    def __init__(self, model_id, torch_dtype="auto", load_in_4bit=False):
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

    @abstractmethod
    def run(self):
        pass


class FullInfoJsonPipeline(MultiModalPipeline):
    def run(self, image_name, save_output=True):
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


class ObjectPropertiesPipeline(MultiModalPipeline):
    def _get_one_object_properties(
        self, image_path: Path, detection_dict: str, system_prompt: str
    ) -> dict[str, str]:
        image = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots()
        ax.imshow(image)
        label, box = detection_dict["label"], detection_dict["box"]
        text = f"Describe properties of this {label} with bounding_box {box}, like in instruction (format YAML):"
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
            xmin,
            ymin,
            label,
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close()
        marked_image = Image.open(img_buf)

        output_text = self._run(
            marked_image, text, system_prompt=system_prompt, max_new_tokens=384
        )
        img_buf.close()
        output_text = (
            output_text.replace(
                "```yaml\n",
                "",
            )
            .replace(
                "```YAML\n",
                "",
            )
            .replace("```", "")
        )
        print(output_text)
        try:
            output_dict = yaml.safe_load(output_text)
        except:
            return {
                "name": label,
                "color": "unknown",
                "form": "unknown",
                "size": "unknown",
                "location": "unknown",
                "purpose": "unknown",
                "relationship": "unknown",
            }
        return output_dict

    def get_objects_properties(
        self, image_path: Path, detections_list
    ) -> list[dict[str, str]]:
        with open(
            os.path.join(PROMPT_DIRECTORY, "system", "object_properties_prompt.txt"),
            encoding="utf-8",
        ) as f:
            system_prompt = f.read()
        properties_list = []
        for detection in detections_list:
            output_dict = self._get_one_object_properties(
                image_path, detection, system_prompt
            )

            properties_list.append(output_dict)
        return properties_list

    def _construct_final_json(self, properties_list, bboxes):
        return {
            "image_name": bboxes["filename"].split("/")[-1],
            "objects": properties_list,
            "2d_bbox": [
                {"object": d["label"], "bbox": d["box"]} for d in bboxes["objects"]
            ],
        }

    def run(self, image_path: Path):
        image_name = image_path.name
        with open(
            os.path.join(str(DETECTIONS_DIR), f"{image_name}.json"), encoding="utf-8"
        ) as file:
            json_of_detections_and_bboxes = json.load(file)
            detections_list = json_of_detections_and_bboxes["objects"]
        # get properties of objects on image
        properties_list = self.get_objects_properties(image_path, detections_list)
        # get full info about image
        main_info_json = self._construct_final_json(
            properties_list, json_of_detections_and_bboxes
        )
        # get description of image
        with open(
            os.path.join(PROMPT_DIRECTORY, "system", "only_description_prompt.txt"),
            encoding="utf-8",
        ) as f:
            system_prompt = f.read()
        image_filepath = os.path.join(DETECTIONS_PICTURES_DIR, f"{image_name}.png")

        def dict_to_string(d):
            result = ""
            for key, value in d.items():
                result += f'{key}: "{value}"\n'
            return result.strip()

        detections_list_str = ""
        for item in detections_list:
            output = dict_to_string(item)
            detections_list_str += output + "\n---\n"
        description = self._run(
            image_filepath,
            detections_list_str,
            system_prompt=system_prompt,
            max_new_tokens=128,
        )
        # if description output is very big - take only part of output
        if len(description.split(".")) > 6:
            description = ".".join(description.split(".")[:6])
        if len(description.split(" ")) > 70:
            description = " ".join(description.split(" ")[:70])

        final_json = {}
        final_json["image_name"] = main_info_json["image_name"]
        final_json["description"] = description
        final_json["objects"] = main_info_json["objects"]
        final_json["2d_bbox"] = main_info_json["2d_bbox"]
        try:
            with open(
                ANNOTATIONS_DIR / f"{image_name}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(final_json, f, indent=4, ensure_ascii=False)
                print("Saved! ", image_name)
        except Exception as ex:
            print("EXCEPTION:\n", image_name, ex)


class GenerateClassesPipeline(MultiModalPipeline):
    def run(self, image_name, input_folder, save_output=True):
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
    # classes_pipeline = GenerateClassesPipeline("Qwen/Qwen2-VL-2B-Instruct")
    properties_pipeline = ObjectPropertiesPipeline("Qwen/Qwen2-VL-2B-Instruct")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(CLASSES_DIR, exist_ok=True)
    image_files = image_folder.get_images_list()
    for image_file in tqdm(image_files):
        # classes_pipeline.run(image_file.name, image_folder.folder_path)
        properties_pipeline.run(image_file)
