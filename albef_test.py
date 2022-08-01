from example_models.albef.engine import ALBEF
from vl_checklist.evaluate import Evaluate


if __name__ == '__main__':
    model = ALBEF('ALBEF.pth')
    eval = Evaluate("configs/sample.yaml", model=model)
    eval.start()
    


