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

from utils import AttributeDict
from utils import xml_utils

from data.labelbox.labelbox_wrapper import LabelBoxWrapper
from data.roboflow.roboflow_wrapper import RoboflowWrapper
from data.config.config_keys import CFG
from data import data_selection

from models.train_model import train_model
from models.predict_model import predict_on_folder


def log(string):
    print(string)


def download_dataset_for_train(config_dict):
    if config_dict.labelbox_enabled:
        # 1.1. Also Convert (all) Data for training to csv
        downloaded_dataset_path = Path("./")
        LabelBoxWrapper.download_dataset(config_dict.labelbox_credentials, at_path=downloaded_dataset_path)
        # 1. Update the config (train/val) csv paths to the ones form the downloaded dataset
        config_dict.train_annotations = downloaded_dataset_path / "train_anno.csv"
        config_dict.valid_annotations = downloaded_dataset_path / "valid_anno.csv"
        
    elif config_dict.roboflow_enabled:
        # 1.1. Also Convert (all) Data for training to csv
        dataset = RoboflowWrapper.download_dataset(config_dict.roboflow_credentials)
        downloaded_dataset_path = Path(dataset.location)
        # 1. Update the config (train/val) csv paths to the ones form the downloaded dataset
        config_dict.train_annotations = downloaded_dataset_path / "train" / "train_anno.csv"
        config_dict.valid_annotations = downloaded_dataset_path / "valid" / "valid_anno.csv"
        xml_utils.xmls_to_csv_for_train(config_dict.train_annotations.parent, config_dict.train_annotations)
        xml_utils.xmls_to_csv_for_train(config_dict.valid_annotations.parent, config_dict.valid_annotations)
        # voc format start from 1 -> 417, while images are 416x416
        RoboflowWrapper.fix_roboflow_specific_issues_on_csv(config_dict.train_annotations)
        RoboflowWrapper.fix_roboflow_specific_issues_on_csv(config_dict.valid_annotations)
        

def convert_predict_output_to_batches_to_be_labeled(config_dict):
    # 1. Data Selection (top k unsure)
    selection_output = Path(config_dict.unlabeled_crops_path).parent / "k_most_unsure"
    data_selection.select_k_most_unreliable_predictions_from_folder(
        k=20,
        from_folder=Path(config_dict.unlabeled_crops_path),
        output_folder=selection_output
    )
    # 1. Move selected crops to labeled folder, while waiting for annotations,
    # and after that, just use them from the same folder.
    # Predictions stay in the output_folder and will be sent for tagging
    data_selection.move_image_files_from_folder_to_folder(selection_output, config_dict.pretagged_crops_path)
    # # 1. Split them into train / val (/test)
    # data_selection.split_files_at_path(selection_output, files_extension=".xml", splits=config_dict.splits)


def upload_the_new_pretagged_batches_from_path(config_dict, from_path=Path("./")):
    if config_dict.labelbox_enabled:
        LabelBoxWrapper.upload_the_new_pretagged_batches_from_path(config_dict.labelbox_credentials, from_path=from_path)
    elif config_dict.roboflow_enabled:
        RoboflowWrapper.upload_the_new_pretagged_batches_from_path(config_dict.roboflow_credentials, from_path=from_path)


def run_train_flow(config_dict):
    # 1. Download (all) Data for training
    log(">>>>>> Download (all) Data for training")
    download_dataset_for_train(config_dict)
    # 1. Train new model
    log(">>>>>> Train new model")
    model_path, eval_report = train_model(config_dict)
    # 1. Evaluate model (on valid); (and check if it is better than prev model)
    # TODO (FOC): MAKE THE CHECK
    # 1. Predict using new model on unlabeled data
    log(">>>>>> Predict using new model on unlabeled data")
    predict_on_folder(config_dict.unlabeled_crops_path, available_gpus=config_dict.gpus)
    # 1. Data Selection (top k unsure); Move selected crops to labeled folder; Split them into train / val (/test)
    log(">>>>>> Data Selection (top k unsure); Move selected crops to labeled folder; Split them into train / val (/test)")
    convert_predict_output_to_batches_to_be_labeled(config_dict)
    # 1. Send data to be labeled by ORACLE as batch
    log(">>>>>> Send data to be labeled by ORACLE as batch")
    upload_the_new_pretagged_batches_from_path(config_dict)
    

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
    run_train_flow(AttributeDict(loaded_config))
