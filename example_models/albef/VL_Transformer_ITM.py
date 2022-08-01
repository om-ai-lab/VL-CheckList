from functools import partial
from example_models.albef.models.vit import VisionTransformer
from example_models.albef.models.xbert import BertConfig, BertModel

import torch
from torch import nn
from torchvision import transforms

class VL_Transformer_ITM(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = ''
                 ):
        super().__init__()
    
        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 



        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)   
        
        self.itm_head = nn.Linear(768, 2) 

        
    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        output = self.text_encoder(text.input_ids.to(image.device), 
                                attention_mask = text.attention_mask.to(image.device),
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts.to(image.device),      
                                return_dict = True,
                               )     
        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings)
        #_,pred = torch.max(vl_output,1)
        return vl_output
