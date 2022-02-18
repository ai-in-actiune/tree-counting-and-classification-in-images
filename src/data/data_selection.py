import os
import argparse
import shutil
import pandas as pd
from pathlib import Path
# internal
from utils.constants import *


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
            output_folder: the folder path of the output: in same structure as above
        Returns:
            Nothing. But moves the k most uncertain predictions (images) to the output folder
    """
    initial_predictions_df = pd.read_csv(from_folder / ALL_PREDICTIONS_CSV)
    image_confidences_df = initial_predictions_df.groupby(by=IMAGE_PATH)[CONFIDENCE_COL].mean()
    image_weak_confidences_df = image_confidences_df.sort_values(ascending=True)[:k]
    output_predictions_df = initial_predictions_df[initial_predictions_df[IMAGE_PATH].apply(lambda x: x in image_weak_confidences_df)]
    
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
    parser.add_argument("-o", "--output_folder", type=str, required=False)
    parser.add_argument("-k", "--k_unsure", type=int, required=True, help="k unsure images to select and move to the 'needs-review' (output) folder")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    k_unsure = args.k_unsure
    
    select_k_most_unreliable_predictions_from_folder(k_unsure,
                                                     from_folder=input_folder, output_folder=output_folder)
