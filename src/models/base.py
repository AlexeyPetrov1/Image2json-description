from abc import ABC, abstractmethod


class NotLoadedModelError(Exception):
    DEFAULT_MESSAGE = "Model not loaded. Please, call `your_model.load()`"

    def __init__(self, message=None):
        self.message = message
        if self.message == None:
            self.message = self.DEFAULT_MESSAGE
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class ModelInterface(ABC):
    @abstractmethod
    def predict(self, inputs: dict) -> dict:
        pass

    @abstractmethod
    def load(self):
        pass
