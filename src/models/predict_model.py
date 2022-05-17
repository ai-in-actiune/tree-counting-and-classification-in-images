import argparse
import json
from glob import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
# external
from deepforest import main
import torch
# internal
from utils import xml_utils
from utils.constants import *

class DeepTreePredictor:
    """
    Given a model, predicts on images.
    Model is currently expected to be from 'deepforest' lib, so current class is a wrapper.
    Apart from wrapping the 'deepforest' predictor, DeepTreePredictor implements some utility methods
    used for fast predict on folder with images.
    
    Outputting the xml/image is useful for tagging afterwards via LabelImg.
    Outputting the csv is useful for fast handling and read/write via pandas.
    
    The output will be in the same given folder, right near each image,
    since the output will be in the same name as the image.
    e.g. folder_X/img1.png folder_X/img2.png
    after calling:  folder_X/img1.png folder_X/img1.csv folder_X/img1.xml
                    folder_X/img2.png folder_X/img2.csv folder_X/img2.xml
    Outputting the csv or the xml is optional. Apart from these outputs, also a csv for the whole
    images is generated under ALL_PREDICTIONS_CSV.
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
        for img_path_str in tqdm(images_paths, desc="Predicting on images"):
            img_path = Path(img_path_str)
            bboxes_df = self.predict_image(img_path_str)
            bboxes_df = bboxes_df if bboxes_df is not None else pd.DataFrame()
            if write_xml:
                xml_utils.annotations_to_xml(bboxes_df, img_path_str, write_file=True)
            if write_csv:
                bboxes_df.to_csv(img_path.parent / f'{img_path.stem}.csv', index=False)
            bboxes_df[IMAGE_PATH] = img_path.name
            accumulator_bboxes_dfs.append(bboxes_df)
        folder_bboxes_df = pd.concat(accumulator_bboxes_dfs)
        folder_bboxes_df.to_csv(Path(folder_path) / ALL_PREDICTIONS_CSV, index=False)
        return folder_bboxes_df


def get_model(model_path=None, checkpoint_path=None, available_gpus=0):
    """
    Loads model at path.
    Loads the labels.json also from that same path.
    model_path can be None, in order to use the default predictor
    """
    if model_path:
        model_path = Path(model_path)
        with open(model_path.parent / "labels.json", 'r') as f:
            labels_dict = json.load(f)
        model = main.deepforest()
        torch_device = torch.device('cpu') if available_gpus == 0 else torch.device('gpu')
        model.model.load_state_dict(torch.load(model_path, map_location=torch_device))
    elif checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path.parent / "labels.json", 'r') as f:
            labels_dict = json.load(f)
        model = main.deepforest.load_from_checkpoint(str(checkpoint_path),
                                                     num_classes=len(labels_dict),
                                                     label_dict=labels_dict)
    else:
        model = main.deepforest()
        model.use_release()
        
    model.config["gpus"] = available_gpus
    
    return model


def predict_on_folder(folder_path):
    dtp = DeepTreePredictor(get_model())
    dtp.predict_on_folder(folder_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-modm", "--model_folder_path", type=str, required=False)
    parser.add_argument("-modc", "--checkpoint_path", type=str, required=False)
    parser.add_argument("-gpus", "--available_gpus", type=int, default=0, required=False)
    parser.add_argument("-c", "--write_csvs", action="store_true",
                        help="Just a flag argument. Where action='store_true' implies default=False.")
    parser.add_argument("-x", "--write_xmls", action="store_true",
                        help="Just a flag argument. Where action='store_true' implies default=False.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # in
    input_folder = args.input_folder
    # model
    model_folder_path = args.model_folder_path
    checkpoint_path =  args.checkpoint_path
    available_gpus = args.available_gpus
    # out
    write_csvs = args.write_csvs
    write_xmls = args.write_xmls

    dtp = DeepTreePredictor(get_model(model_path=model_folder_path, checkpoint_path=checkpoint_path, available_gpus=available_gpus))
    dtp.predict_on_folder(input_folder, write_csv=write_csvs, write_xml=write_xmls)
