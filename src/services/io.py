from src.config.params import PROMPT_DIRECTORY
import os

def get_default_classes():
    with open(os.path.join(PROMPT_DIRECTORY, 'user', "default_humanoid_robot_classes.txt"), encoding="utf-8") as f:
        text_classes = f.read()
        classes = list(set(map(str.strip, text_classes.split("."))))
        text_classes = '.'.join(classes)
    return text_classes, classes