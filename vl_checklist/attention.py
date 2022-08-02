import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from example_models.albef.engine import ALBEF
from example_models.vilt.engine import ViLT
from example_models.vilt.HeatMap import HeatMap
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from PIL import Image
from example_models.vilt.modules.objectives import cost_matrix_cosine, ipot


class Attention:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_attention_by_heatmap(self, infer, image, tokenizer, input_ids):
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
        for hidx in range(1, len(input_ids) - 1):
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
            hm.plot(title=selected_token)

            outputs.append((hm, selected_token))

    def get_attention_by_gradcam(self, model, tokenizer, image_path, image_input, text_input, attr_name, target_layer):
        encoder_name = getattr(model, attr_name, None)
        encoder_name.encoder.layer[target_layer].crossattention.self.save_attention = True
        output = model(image_input, text_input)
        loss = output[:, 1].sum()

        model.zero_grad()
        loss.backward()
        image_size = 256
        temp = int(np.sqrt(image_size))

        # the effect of mask is let those padding tokens multiply with 0 so that they won't be calculated in cams and
        # grads , because of the text preprocess of ALBEF and TCL, mask is unuseful here
        mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1)
        grads = encoder_name.encoder.layer[target_layer].crossattention.self.get_attn_gradients()
        cams = encoder_name.encoder.layer[target_layer].crossattention.self.get_attention_map()

        cams = cams[:, :, :, 1:].reshape(image_input.size(0), 12, -1, temp, temp) * mask
        grads = grads[:, :, :, 1:].clamp(0).reshape(image_input.size(0), 12, -1, temp, temp) * mask

        gradcam = cams * grads
        gradcam = gradcam[0].mean(0).cpu().detach()

        num_image = len(text_input.input_ids[0]) + 1
        fig, ax = plt.subplots(nrows=num_image, ncols=1, figsize=(12, 10 * num_image))
        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        ax[0].imshow(rgb_image)
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel("Image")

        for i, token_id in enumerate(text_input.input_ids[0][:]):
            word = tokenizer.decode([token_id])
            gradcam_image, attMapV = self.gradcam_postprocess(rgb_image, gradcam[i])
            ax[i + 1].imshow(gradcam_image)
            ax[i + 1].set_yticks([])
            ax[i + 1].set_xticks([])
            ax[i + 1].set_xlabel(word)

        plt.show()

    def get_attention_by_bbox(self):
        pass

    def gradcam_postprocess(self, img, attMap, blur=True, overlap=True):
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        # resize the attMap to match the size of origin image
        # img.shape[:2] means the height and width of origin image, img.shape[2] is the num of channels (it is 3 here)
        attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()
        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
                attMap.shape + (1,)) * attMapV
        return attMap, attMapV

    def getAttMap(self, image_path, text):
        if self.model_name.lower() == 'albef':
            engine = ALBEF('ALBEF_4M.pth')
            model, tokenizer = engine.load_model(engine.model_id)
            image_input = engine.load_data(src_type='local', data=[image_path])[0]
            text_input = tokenizer(engine.pre_caption(text), return_tensors="pt")
            self.get_attention_by_gradcam(model, tokenizer, image_path, image_input, text_input,
                                          attr_name='text_encoder', target_layer=8)
        elif self.model_name.lower() == 'vilt':
            engine = ViLT('vilt_200k_mlm_itm.ckpt')
            model, tokenizer = engine.load_model(engine.model_id)
            image_input = engine.load_data(src_type='local', data=[image_path])[0]
            encoded = tokenizer([text], padding='longest')
            batch = {'text': [text], 'image': [image_input],
                     'text_ids': torch.tensor(encoded['input_ids']).to(engine.device),
                     "text_labels": torch.tensor(encoded["input_ids"]).to(engine.device),
                     "text_masks": torch.tensor(encoded["attention_mask"]).to(engine.device)}
            infer = model(batch)
            self.get_attention_by_heatmap(infer, image_path, tokenizer, encoded['input_ids'][0])


if __name__ == '__main__':
    attention = Attention('albef')
    attention.getAttMap(image_path='',
                        text='')
