import torch
import requests
import io
import os
from PIL import Image
from typing import List
import base64
import json
from example_models.vilt.modules.vilt_module import ViLTransformer
from example_models.vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from example_models.vilt.transforms import pixelbert_transform
from example_models.utils.helpers import LRUCache, chunks
from example_models.vilt.modules.objectives import cost_matrix_cosine, ipot
import numpy as np
from example_models.vilt.HeatMap import HeatMap

from vl_checklist.vlp_model import VLPModel


class ViLT(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../")
    MAX_CACHE = 20

    def __init__(self, model_id):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 16
        self.device = "cuda:0"
        self.model_dir = "resources"
        self.model_id = model_id
    
    def model_name(self):
        return self.model_id

    def _load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")

        if not self._models.has(model_id):
            _config = {"exp_name": "vilt", "seed": 0, "datasets": ["coco", "vg", "sbu", "gcc"], "loss_names": {"itm": 1, "mlm": 0, "mpp": 0, "vqa": 0, "nlvr2": 0, "irtr": 0}, "batch_size": 4096, "train_transform_keys": ["pixelbert"], "val_transform_keys": ["pixelbert"], "image_size": 384, "max_image_len": 200, "patch_size": 32, "draw_false_image": 1, "image_only": False, "vqav2_label_size": 3129, "max_text_len": 40, "tokenizer": "bert-base-uncased", "vocab_size": 30522, "whole_word_masking": False, "mlm_prob": 0.15, "draw_false_text": 0, "vit": "vit_base_patch32_384", "hidden_size": 768, "num_heads": 12, "num_layers": 12, "mlp_ratio": 4, "drop_rate": 0.1, "optim_type": "adamw", "learning_rate": 0.0001, "weight_decay": 0.01, "decay_power": 1, "max_epoch": 100, "max_steps": 25000, "warmup_steps": 2500, "end_lr": 0, "lr_mult": 1, "get_recall_metric": False, "resume_from": None, "fast_dev_run": False, "val_check_interval": 1.0, "test_only": True, "data_root": "", "log_dir": "result", "per_gpu_batchsize": 0, "num_gpus": 1, "num_nodes": 1, "load_path": "", "num_workers": 8, "precision": 16}
            _config['load_path'] = os.path.join(self.root_dir, self.model_dir, model_id)
            tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
            model = ViLTransformer(_config)
            model.setup("test")
            model.eval()
            model.to(self.device)
            self._models.put(model_id, (model, tokenizer))

        return self._models.get(model_id)

    def _load_data(self, src_type, data):
        def transform(x):
            img = x.resize((384, 384))
            img = pixelbert_transform(size=384)(img)
            img = img.unsqueeze(0).to(self.device)
            return img

        if src_type == 'local':
            image_data = []
            for x in data:
                temp = Image.open(x).convert('RGB')
                image_data.append(transform(temp))

        elif src_type == 'url':
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(requests.get(x).content)).convert("RGB")
                image_data.append(transform(temp))

        elif src_type == 'base64':
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(base64.b64decode(x)))
                image_data.append(transform(temp))
        else:
            raise Exception("Unknown mode {}.".format(src_type))

        return image_data

    def predict(self, image_paths: List[str],
                texts: List[str],
                src_type: str = 'local'):        
        
        if not len(texts) == len(image_paths):
            raise Exception("# of texts and # of images should be matched")

        model, tokenizer = self._load_model(self.model_id)
        # process images by batch
        probs = []
        logits = []

        for chunk_i, chunk_t in zip(chunks(image_paths, self.batch_size), chunks(texts, self.batch_size)):
            image_data = self._load_data(src_type, chunk_i)

            batch_images = []  # (num_image x num_text)
            batch_text = []

            for i,t in zip(image_data,chunk_t):
                batch_images.append(i)
                batch_text.append(t)

            batch = {"text": batch_text, "image": batch_images}

            inferred_token = batch_text
            batch["text"] = inferred_token
            encoded = tokenizer(inferred_token, padding='longest')
                        
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(self.device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(self.device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(self.device)
            
            with torch.no_grad():
                infer = model(batch)            

                itm_logits = model.itm_score(infer["cls_feats"])
                soft_prob = torch.softmax(itm_logits, dim=1)

                probs.extend(soft_prob.tolist())
                #logits.extend(itm_logits.tolist())
        
        return {"probs":probs} # {'probs': [[0.00455933902412653, 0.9954406023025513], [0.999612033367157, 0.00038797641173005104], [0.9999412298202515, 5.878580122953281e-05]]}
    
    def generate_heatmap(self, infer, image, tokenizer, input_ids):
        image = Image.open(image).convert('RGB')
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = (
            infer["text_masks"].bool(),
            infer["image_masks"].bool(),
        )
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)
        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
                    dtype=cost.dtype
                )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
                    dtype=cost.dtype
                )
        T = ipot(
                    cost.detach(),
                    txt_len,
                    txt_pad,
                    img_len,
                    img_pad,
                    joint_pad,
                    0.1,
                    1000,
                    1,
                )
        
        plan = T[0]
        plan_single = plan * len(txt_emb)
        outputs = []
        for hidx in range(1,len(input_ids)-1):
            cost_ = plan_single.t()
            cost_ = cost_[hidx][1:].cpu()

            patch_index, (H, W) = infer["patch_index"]
            heatmap = torch.zeros(H, W)
            for i, pidx in enumerate(patch_index[0]):
                h, w = pidx[0].item(), pidx[1].item()
                heatmap[h, w] = cost_[i]

            heatmap = (heatmap - heatmap.mean()) / heatmap.std()
            heatmap = np.clip(heatmap, 1.0, 3.0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            selected_token = tokenizer.convert_ids_to_tokens(
                    input_ids[hidx]
                )        
            if not torch.isnan(heatmap).any():
                hm = HeatMap(image, heatmap.cpu().numpy())
            else:
                heatmap = np.zeros(heatmap.shape)
                hm = HeatMap(image, heatmap)
            
            outputs.append((hm,selected_token))
        return outputs



