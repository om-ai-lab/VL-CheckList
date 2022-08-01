import torch
import requests
import io
import os
from PIL import Image
from typing import List
import base64
import re
from torchvision import transforms
from example_models.albef.VL_Transformer_ITM import VL_Transformer_ITM
from example_models.albef.models.tokenization_bert import BertTokenizer
from example_models.utils.helpers import LRUCache, chunks
from vl_checklist.vlp_model import VLPModel

class ALBEF(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../")
    MAX_CACHE = 20

    def __init__(self, model_id):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 16
        self.device = "cuda"
        self.model_dir = "resources"
        self.model_id = model_id
    
    def model_name(self):
        return "ALBEF"

    def load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")

        if not self._models.has(model_id):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=os.path.join(os.path.dirname(os.path.abspath(__file__)),'config_bert.json'))
            checkpoint = torch.load(os.path.join(self.root_dir, self.model_dir, model_id), map_location='cpu')              
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint = {k.replace('.bert', ''): v for k, v in checkpoint.items()}

            msg = model.load_state_dict(checkpoint,strict=False)
            model.eval()
            model.to(self.device)
            self._models.put(model_id, (model, tokenizer))

        return self._models.get(model_id)



    def load_data(self, src_type, data):
        dim = 256 # prev=384
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        transform_albef = transforms.Compose([
            transforms.Resize((dim,dim),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        if src_type == 'local':
            image_data = []
            for x in data:
                temp = Image.open(x).convert('RGB')
                image_data.append(transform_albef(temp).unsqueeze(0).to(self.device))

        elif src_type == 'url':
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(requests.get(x).content)).convert("RGB")
                image_data.append(transform_albef(temp).unsqueeze(0).to(self.device))

        elif src_type == 'base64':
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(base64.b64decode(x)))
                image_data.append(transform_albef(temp).unsqueeze(0).to(self.device))
        else:
            raise Exception("Unknown mode {}.".format(src_type))

        return image_data

    def pre_caption(self,caption,max_words=30):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n') 
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words)>max_words:
            caption = ' '.join(caption_words[:max_words])            
        return caption

    def predict(self, images: List[str],
                texts: List[str],
                src_type: str = 'local'
                ):

        model, tokenizer = self.load_model(self.model_id)

        # process images by batch
        probs = []        
        for chunk_i, chunk_t in zip(chunks(images, self.batch_size), chunks(texts, self.batch_size)):
            image_data = self.load_data(src_type, chunk_i)

            batch_images = []  # (num_image x num_text)
            batch_text = []

            for i,t in zip(image_data,chunk_t):
                t = self.pre_caption(t)
                text_input = tokenizer(t, return_tensors="pt")
                batch_images.append(i)
                batch_text.append(text_input)
            with torch.no_grad():
                for image,text in zip(batch_images,batch_text):            
                    itm_logits = model(image, text)                    
                    soft_prob = torch.softmax(itm_logits, dim=1)                    
                    probs.extend(soft_prob.tolist())
                    
        return {"probs":probs}
