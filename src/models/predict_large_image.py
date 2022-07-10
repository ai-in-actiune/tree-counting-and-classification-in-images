import argparse
from deepforest import get_data
from deepforest import preprocess

import torch
from torchvision.ops import nms
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.plotting import plot_bboxes
from models.predict_model import get_model


def predict_large_image(
    numpy_image, model, patch_size=400, patch_overlap=0.05, iou_threshold=0.1, should_display=False
):
    # crop original image
    windows = preprocess.compute_windows(
        numpy_image=numpy_image, patch_size=patch_size, patch_overlap=patch_overlap
    )

    predictions_list = []

    for window in tqdm(windows, desc=f"Iterating sliding patches", disable=(not should_display)):
        crop = numpy_image[window.indices()]
        pred_df = model.predict_image(crop.astype(np.float32))

        # adjust crop offset relative to the original image
        pred_df["xmin"] = window.x + pred_df["xmin"]
        pred_df["ymin"] = window.y + pred_df["ymin"]
        pred_df["xmax"] = (window.x + window.w) - (patch_size - pred_df["xmax"])
        pred_df["ymax"] = (window.y + window.h) - (patch_size - pred_df["ymax"])

        predictions_list.append(pred_df)

    final_preds_df = pd.concat(predictions_list)
    final_preds_df.reset_index(drop=True, inplace=True)

    # filter out duplicate bboxes as a result of crops overlap by applying NMS
    bboxes = final_preds_df[["xmin", "ymin", "xmax", "ymax"]].values
    confs = final_preds_df["score"].values

    ixs = nms(torch.tensor(bboxes), torch.tensor(confs), iou_threshold=iou_threshold)
    bboxes = bboxes[ixs]
    bboxes_df = pd.DataFrame(bboxes, columns=["xmin", "ymin", "xmax", "ymax"])

    numpy_image = plot_bboxes(numpy_image.copy(), bboxes_df, (59, 255, 242),
                              DISPLAY=should_display)
    return numpy_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image_path', type=str, required=True)
    args = parser.parse_args()

    sample_image = get_data(args.image_path)
    image = cv.imread(sample_image)[..., ::-1]

    predict_large_image(image, model=get_model(), patch_overlap=0.25)
