from pathlib import Path
import torch
import json
# external
from deepforest import main


def get_model(model_path=None, available_gpus=0):
    """
    Loads model at path.
    Loads the labels.json also from that same parent path.
    model_path can be None, in order to use the default predictor
    """
    is_checkpoint_path = Path(model_path).lower().endswith(('.pkl', '.pl'))
    if model_path:
        if is_checkpoint_path:
            model_path = Path(model_path)
            with open(model_path.parent / "labels.json", 'r') as f:
                labels_dict = json.load(f)
            model = main.deepforest.load_from_checkpoint(str(model_path),
                                                         num_classes=len(labels_dict),
                                                         label_dict=labels_dict)
        else:
            model_path = Path(model_path)
            with open(model_path.parent / "labels.json", 'r') as f:
                labels_dict = json.load(f)
            model = main.deepforest()
            torch_device = torch.device('cpu') if available_gpus == 0 else torch.device('gpu')
            model.model.load_state_dict(torch.load(model_path, map_location=torch_device))
    else:
        model = main.deepforest()
        model.use_release()
    model.config["gpus"] = available_gpus
    
    return model
