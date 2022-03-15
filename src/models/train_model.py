import os
from glob import glob
import argparse
from pathlib import Path
import yaml

import pandas as pd
from deepforest import main
from tqdm import tqdm

from utils.xml_utils import xml_to_annotations


def extract_labels_as_csv(from_folder_path, to_file):
    xml_paths = sorted(glob(f"{from_folder_path}/*.xml"))

    accumulator_bboxes_dfs = []
    for xml_path_str in tqdm(xml_paths, desc="Converting xml files to csv"):
        xml_path = Path(xml_path_str)
        xml_as_df = xml_to_annotations(str(xml_path))
        accumulator_bboxes_dfs.append(xml_as_df)

    folder_bboxes_df = pd.concat(accumulator_bboxes_dfs)
    folder_bboxes_df.to_csv(to_file, index=False)


def train_model(
    train_annotations: Path,
    valid_annotations: Path,
    config_path: Path,
    output_path: Path,
):
    """
    Args:
        train_annotations: path to annotations file for training
        valid_annotations: path to annotations file for validation
        config_path: path to hyper-parameters config file
        output_path: path to save the trained model and evaluation report
    Returns:
        Nothing. It saves the model and evaluation report
    """
    loaded_config = yaml.safe_load(open(config_path))

    m = main.deepforest()
    m.use_release()
    m.config["epochs"] = loaded_config["epochs"]
    m.config["batch_size"] = loaded_config["batch_size"]
    m.config["save-snapshot"] = loaded_config["save-snapshot"]
    m.config["train"]["csv_file"] = str(train_annotations)
    m.config["train"]["root_dir"] = str(train_annotations.parent)
    m.config["validation"]["csv_file"] = str(valid_annotations)
    m.config["validation"]["root_dir"] = str(valid_annotations.parent)

    m.create_trainer()
    m.trainer.fit(m)
    eval_report = m.evaluate(
        csv_file=m.config["validation"]["csv_file"],
        root_dir=m.config["validation"]["root_dir"],
    )

    os.makedirs(output_path, exist_ok=True)
    eval_report_df = eval_report["results"]
    eval_report_df.to_csv(f"{output_path}/report.csv", index=False)
    m.trainer.save_checkpoint(f"{output_path}/checkpoint.pl")


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

    train_model(
        train_annotations=train_csv_path,
        valid_annotations=valid_csv_path,
        config_path=config_path,
        output_path=output_path,
    )
