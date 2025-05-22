from typing import Tuple


class BoxFilter:
    def __init__(self, min_area_ratio: float):
        self.min_area_ratio = min_area_ratio

    def is_big_enough(self, bbox: Tuple[int, int, int, int], image) -> bool:
        image_width, image_height = image.size
        image_area = image_width * image_height
        x_min, y_min, x_max, y_max = bbox
        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area / image_area >= self.min_area_ratio:
            return True
        return False
