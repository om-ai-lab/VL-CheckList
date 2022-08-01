from example_models.vilt.engine import ViLT
from vl_checklist.evaluate import Evaluate


if __name__ == '__main__':
    h_vilt = ViLT('vilt_200k_mlm_itm.ckpt')
    vilt_eval = Evaluate("configs/sample.yaml", model=h_vilt)
    vilt_eval.start()
    


