import os
from vl_checklist.vlp_model import VLPModel
from example_models.utils.helpers import LRUCache, chunks
import torch.cuda
from open_clip import create_model_and_transforms, trace_model, get_tokenizer
import logging
import fsspec

from PIL import Image


# TODO: this is repeated in the other file too (src/training/file_utils.py)
def pt_load(file_path, map_location=None):
    if not file_path.startswith('/'):
        logging.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out



class OpenCLIP(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    MAX_CACHE = 20

    def __init__(self,model_id):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "resources"
        self.model_id = model_id
        self.model, self.preprocess = self._load_model(self.model_id)

    def model_name(self):
        return self.model_id

    def _load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")
        print("Loading model: {}".format(model_id))

        model, preprocess_train, preprocess_val = create_model_and_transforms("ViT-B-32", "openai", device= self.device)

        checkpoint = pt_load(model_id, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
    
        return model, preprocess_val

    def _load_data(self, src_type, data):
        pass

    def predict(self,
                images: list,
                texts: list,
                src_type: str = 'local'
                ):
        # process images by batch
        probs = []
        for i, chunk_i in enumerate(chunks(images, self.batch_size)):
            for j in range(len(chunk_i)):
                try:
                    image = self.preprocess(Image.open(chunk_i[j])).unsqueeze(0).to(self.device)
                except Exception as e:
                    print(e)
                    continue
                
                # text format is [["there is a cat","there is a dog"],[...,...]...]
                tokenizer=get_tokenizer("ViT-B-32")
                text = tokenizer(texts[j]).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    text_features = self.model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    logits_per_image = image_features @ text_features.T
                    prob = logits_per_image.item()
                    probs.append(prob)
    
        return {"probs":probs}
        


