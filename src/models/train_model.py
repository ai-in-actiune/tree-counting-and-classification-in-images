import os
from glob import glob
import argparse
from pathlib import Path
import yaml
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from models.deep_tree_model import get_model, save_model
from data.data_selection import extract_labels_as_csv
from data.neptune_ai.neptune_wrapper import NeptuneWrapper
from data.config.config_keys import CFG


def train_model_w_params(config_path: Path, **kwargs):
    """
    Args:
        config_path: path to hyper-parameters config file
        kwargs: optional in order to use instead of the values from config_path
            train_annotations: path to annotations file for training
            valid_annotations: path to annotations file for validation
            output_path: path to save the trained model and evaluation report
    Returns:
        It saves the model and evaluation report and returns the trained model_file_path and eval report
    """
    loaded_config = yaml.safe_load(open(config_path))
    loaded_config.update(kwargs)
    return train_model(loaded_config)


def train_model(config_dict):
    """
    Args:
        train_annotations: path to annotations file for training
        valid_annotations: path to annotations file for validation
        config_dict: hyper-parameters and paths config dict
    Returns:
        It saves the model and evaluation report and returns the trained model_file_path and eval report
    """
    # setup model
    m = get_model(model_path=config_dict[CFG.pretrained_path], available_gpus=config_dict[CFG.gpus])
    m.config["epochs"] = config_dict[CFG.epochs]
    m.config["train"]["epochs"] = config_dict[CFG.epochs]
    m.config["batch_size"] = config_dict[CFG.batch_size]
    m.config["save-snapshot"] = config_dict[CFG.save_snapshot]
    m.config["train"]["csv_file"] = str(config_dict[CFG.train_annotations])
    m.config["train"]["root_dir"] = config_dict.get(CFG.crops_path, str(Path(config_dict.train_annotations).parent))
    m.config["validation"]["csv_file"] = str(config_dict[CFG.valid_annotations])
    m.config["validation"]["root_dir"] = config_dict.get(CFG.crops_path, str(Path(config_dict.valid_annotations).parent))
    # prepare train
    m.create_trainer()
    if config_dict[CFG.neptune_enabled]:
        neptune_logger = NeptuneWrapper.get_pytorch_lightning_logger(proj_name=config_dict[CFG.neptune_proj_name],
                                                                     api_token=config_dict[CFG.neptune_api_token])
        m.trainer.logger = neptune_logger
    # train
    m.trainer.fit(m)
    # evaluate
    eval_report = m.evaluate(
        csv_file=m.config["validation"]["csv_file"],
        root_dir=m.config["validation"]["root_dir"],
    )
    # save training results
    output_path = config_dict[CFG.model_output_folder_path]
    os.makedirs(output_path, exist_ok=True)
    eval_report_df = eval_report["results"]
    eval_report_df.to_csv(output_path / "report.csv", index=False)
    # save checkpoint
    m.trainer.save_checkpoint(output_path / "checkpoint.pkl")
    # save model @ torch
    precision = eval_report["box_precision"] * 100
    recall = eval_report["box_recall"] * 100
    time_identifier = datetime.now().strftime("y%Ym%md%dh%H")
    trained_model_name = f'{time_identifier}_deeptree_precision{precision:.2f}_recall{recall:.2f}'
    save_model(m, at_folder_path=output_path, model_name=trained_model_name)
    return output_path / trained_model_name, eval_report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_csv_path", type=str, required=True)
    parser.add_argument("-v", "--valid_csv_path", type=str, required=True)
    parser.add_argument("-c", "--config_path")
    parser.add_argument("-o", "--output_path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_csv_path = Path(args.train_csv_path)
    valid_csv_path = Path(args.valid_csv_path)
    config_path = Path(args.config_path)
    output_path = Path(args.output_path)

    extract_labels_as_csv(train_csv_path.parent, train_csv_path)
    extract_labels_as_csv(valid_csv_path.parent, valid_csv_path)

    train_model_w_params(
        train_annotations=train_csv_path,
        valid_annotations=valid_csv_path,
        config_path=config_path,
        model_output_folder_path=output_path
    )
