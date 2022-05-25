# Input:
# - prev pretrained_model_path & cfg
# - output_folder_path
# - dataset (all raw crops) path
# - neptune cfg
# - labelbox cfg

# 1. Download (all) Data for training
# 1. Train new model
# 1. Evaluate model (on valid); (and check if it is better than prev model)
# 1. Predict using new model on unlabeled data
# 1. Data Selection (top k unsure)
# 1. Send data to be labeled to/by ORACLE as batch

import argparse
import yaml
from pathlib import Path

from data.labelbox.labelbox_wrapper import LabelBoxWrapper
from data.config.config_keys import CFG
from models.train_model import train_model
from models.predict_model import predict_on_folder
from data import data_selection


def convert_predict_output_to_batches_to_be_labeled(config_dict):
    # 1. Data Selection (top k unsure)
    selection_output = Path(config_dict[CFG.unlabeled_crops_path]).parent / "k_most_unsure"
    data_selection.select_k_most_unreliable_predictions_from_folder(
        k=20,
        from_folder=Path(config_dict[CFG.unlabeled_crops_path]),
        output_folder=selection_output
    )
    # 1. Move selected crops to labeled folder, while waiting for annotations,
    # and after that, just use them from the same folder.
    # Predictions stay in the output_folder and will be sent for tagging
    data_selection.move_image_files_from_folder_to_folder(selection_output, config_dict[CFG.labeled_crops_path])
    # 1. Split them into train / val (/test)
    data_selection.split_files_at_path(selection_output, files_extension=".xml", splits=config_dict[CFG.splits])


def run_train_flow(config_dict):
    # 1. Download (all) Data for training
    if config_dict[CFG.labelbox_enabled]:
        # 1.1. Convert (all) Data for training to csv
        LabelBoxWrapper.download_dataset(config_dict[CFG.labelbox_credentials], at_path=Path("./"))
        # 1. Update the config (train/val) csv paths to the ones form the downloaded dataset
        config_dict[CFG.train_annotations] = ""
        config_dict[CFG.valid_annotations] = ""
    # 1. Train new model
    model_path, eval_report = train_model(config_dict)
    # 1. Evaluate model (on valid); (and check if it is better than prev model)
    # TODO (FOC): MAKE THE CHECK
    # 1. Predict using new model on unlabeled data
    predict_on_folder(config_dict[CFG.unlabeled_crops_path], available_gpus=config_dict[CFG.gpus])
    # 1. Data Selection (top k unsure); Move selected crops to labeled folder; Split them into train / val (/test)
    convert_predict_output_to_batches_to_be_labeled(config_dict)
    # 1. Send data to be labeled to/by ORACLE as batch
    if config_dict[CFG.labelbox_enabled]:
        LabelBoxWrapper.upload_the_new_tagging_batches(config_dict[CFG.labelbox_credentials], from_path=Path("./"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained_model_path", type=str, required=True)
    parser.add_argument("-cfg", "--config_path", type=str, required=True)
    parser.add_argument("-o", "--model_output_folder_path", type=str, required=False,
                        help="If not set, it will output in pretrained_model_path's parent folder")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    pretrained_model_path = Path(args.pretrained_model_path)
    config_path = Path(args.config_path)
    output_folder_path = Path(args.model_output_folder_path) \
        if args.model_output_folder_path else Path(pretrained_model_path).parent
    
    loaded_config = yaml.safe_load(open(config_path))
    loaded_config.update({CFG.pretrained_path: pretrained_model_path,
                          CFG.model_output_folder_path: output_folder_path})
    
    run_train_flow(loaded_config)
