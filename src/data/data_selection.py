import os
import argparse
from typing import Dict
import shutil
import random
import math
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from glob import glob
# internal
from utils.constants import *
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


def split_files_at_path(folder_path: Path, files_extension=".xml", splits: Dict[str, float]={"train":0.8, "valid":0.2}):
    files_paths = sorted(glob(f"{folder_path}/*{files_extension}"))
    random.shuffle(files_paths)
    prev_split_index = 0
    for split_name, split_ratio in splits.items():
        split_folder = folder_path / split_name
        os.makedirs(split_folder, exist_ok=True)
        split_end_index = prev_split_index + math.ceil(len(files_paths) * split_ratio)
        for file in files_paths[prev_split_index:split_end_index]:
            shutil.move(file, split_folder / file.name)
        prev_split_index = split_end_index
            

def move_image_files_from_folder_to_folder(from_folder: Path, output_folder: Path):
    for img_path in tqdm(list(from_folder.iterdir()), desc="Moving crops to labeled folder"):
        if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.move(img_path, output_folder / img_path.name)
            

def select_k_most_unreliable_predictions_from_folder(k: int, from_folder: Path, output_folder: Path):
    """
        Selects k most unreliable predictions (images) from given input (from) folder.
        And moves them to the output folder.
        It assumes that the input folder contains at least:
            - [IMG_NAME_0].png
            - [IMG_NAME_0].xml
            - [IMG_NAME_1].png
            - [IMG_NAME_1].xml
            - all_predictions.csv
        Where all_predictions is a csv of all the predictions for all of the images in the folder.
        Usually it is the output of running the predict_model.py / predict_model.sh
        
        The << k most unreliable predictions (images) >> are selected as a mean of all predictions confidences on each image.
        
        Args:
            k: how many of the most unreliable to select
            from_folder: the folder path of the input described above
            output_folder: the folder path of the output: in same structure as above.
                           If folder does not exist, it is created.
        Returns:
            Nothing. But moves the k most uncertain predictions (images) to the output folder
    """
    initial_predictions_df = pd.read_csv(from_folder / ALL_PREDICTIONS_CSV)
    image_confidences_df = initial_predictions_df.groupby(by=IMAGE_PATH)[CONFIDENCE_COL].mean()
    image_weak_confidences_df = image_confidences_df.sort_values(ascending=True)[:k]
    tqdm.pandas(desc=f"Selecting {k} weakest confidence images")
    weakest_confidences_images_mask = initial_predictions_df[IMAGE_PATH].progress_apply(lambda x: x in image_weak_confidences_df)
    output_predictions_df = initial_predictions_df[weakest_confidences_images_mask]
    remaining_initial_predictions_df = initial_predictions_df[~weakest_confidences_images_mask]
    
    # update the remaining predictions_df
    remaining_initial_predictions_df.to_csv(from_folder / ALL_PREDICTIONS_CSV, index=False)
    
    # make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_predictions_df.to_csv(output_folder / ALL_PREDICTIONS_CSV, index=False)
    
    for img_path in image_weak_confidences_df.keys():
        shutil.move(from_folder / img_path, output_folder / img_path)
        xml_path = (from_folder / img_path).stem + '.xml'
        shutil.move(from_folder / xml_path, output_folder / xml_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-k", "--k_unsure", type=int, required=True, help="k unsure images to select and move to the 'needs-review' (output) folder")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    k_unsure = int(args.k_unsure)
    
    select_k_most_unreliable_predictions_from_folder(k_unsure,
                                                     from_folder=input_folder, output_folder=output_folder)
