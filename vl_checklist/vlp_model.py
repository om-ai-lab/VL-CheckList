
from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class VLPModel(ABC): 
   
    @abstractmethod
    def predict(self, image_paths: List[str],
                texts: List[str],
                src_type: str = 'local'
                ):
        pass

    def model_name(self):
        return "VLPModel"
