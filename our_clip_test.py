import os
from example_models.open_clip.engine import OpenCLIP
from vl_checklist.evaluate import Evaluate

CHECKPOINT_DIR = "/network/projects/aishwarya_lab/checkpoints/compositional_vl/Outputs"
CHECKPOINT_NAME = "clip_coco-1e-06-weight0.2/checkpoints/epoch_5.pt"

checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
if __name__ == '__main__':
    model = OpenCLIP(checkpoint_path)
    eval = Evaluate("configs/our_clip.yaml", model=model)
    eval.start()
    


