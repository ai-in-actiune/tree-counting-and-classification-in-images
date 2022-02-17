import argparse
import json
from glob import glob
import pandas as pd
from pathlib import Path
# external
from deepforest import main
# internal
from utils import xml_utils
from utils.constants import *

class DeepTreePredictor:
    """
    Given a model, predicts on images.
    model is currently expected to be from 'deepforest'
    """
    
    def __init__(self, model) -> None:
        self.model = model
    
    def predict_image(self, image_path):
        bboxes = self.model.predict_image(path=image_path)
        return bboxes

    def predict_on_folder(self, folder_path, write_csv=False, write_xml=False):
        """
        Predicts on all images in folder
        
        Args:
            folder_path: where is the folder to run on.
            write_csv: Bool, weather it should write the csv for each image near that image, or not
                       DataFrame with format <detection bbox, label, confidence>
            write_xml: Writes the xml at the same path as the image it describes
        Returns:
            DataFrame with format <image_path, detection bbox, label, confidence>
        """
        images_paths = sorted(glob(f"{str(folder_path)}/*.png"))
        accumulator_bboxes_dfs = []
        for img_path_str in images_paths:
            img_path = Path(img_path_str)
            bboxes_df = self.predict_image(img_path_str)
            
            if write_xml:
                xml_utils.annotations_to_xml(bboxes_df, img_path_str, write_file=True)
            if write_csv:
                bboxes_df.to_csv(img_path.parent / f'{img_path.stem}.csv', index=False)
            bboxes_df[IMAGE_PATH] = img_path.name
            accumulator_bboxes_dfs.append(bboxes_df)
        folder_bboxes_df = pd.concat(accumulator_bboxes_dfs)
        folder_bboxes_df.to_csv(Path(folder_path) / ALL_PREDICTIONS_CSV, index=False)
        return folder_bboxes_df


def get_model(model_folder_path=None):
    """
    Loads model at path.
    Loads the labels.json also from that same path.
    model_folder_path can be None, in order to use the default predictor
    """
    if model_folder_path:
        folder_path = Path(model_folder_path)
        pathmodel = folder_path / "checkpoint.pl"
        with open(folder_path / "labels.json", 'r') as f:
            labels_dict = json.load(f)
        model = main.deepforest.load_from_checkpoint(str(pathmodel),
                                                     num_classes=len(labels_dict),
                                                     label_dict=labels_dict)
    else:
        model = main.deepforest()
        model.use_release()
    return model


def predict_on_folder(folder_path):
    dtp = DeepTreePredictor(get_model())
    dtp.predict_on_folder(folder_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-m", "--model_folder_path", type=str, required=False)
    parser.add_argument("-c", "--write_csvs", action="store_true",
                        help="Just a flag argument. Where action='store_true' implies default=False.")
    parser.add_argument("-x", "--write_xmls", action="store_true",
                        help="Just a flag argument. Where action='store_true' implies default=False.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_folder = args.input_folder
    model_folder_path = args.model_folder_path
    write_csvs = args.write_csvs
    write_xmls = args.write_xmls
    
    dtp = DeepTreePredictor(get_model(model_folder_path=model_folder_path))
    dtp.predict_on_folder(input_folder, write_csv=write_csvs, write_xml=write_xmls)
