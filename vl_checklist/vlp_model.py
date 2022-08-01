
from abc import ABC, abstractmethod
from typing import List


class VLPModel(ABC): 
    @abstractmethod
    def _load_model(self, model_id: str):
        pass

    @abstractmethod
    def _load_data(self, src_type: str, data: List):
        pass
    
    @abstractmethod
    def predict(self, model_id,
                image_paths: List[str],
                texts: List[str],
                src_type: str = 'local'
                ):
        pass

    def model_name(self):
        return "VLPModel"
