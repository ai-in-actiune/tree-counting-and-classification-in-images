import rtree
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np


def create_rtree_from_polygon(coords_list):
    """
    Args:
        coords_list: tuple with coordinates of a polygon
    Returns:
        index: indexed bounding boxes
    """
    index = rtree.index.Index(interleaved=True)
    for coords_idx, coords in enumerate(coords_list):
        index.insert(coords_idx, coords)

    return index


def get_inter_area(box_a, box_b):
    """
    Args:
        box_a: (xmin, ymin, xmax, ymax)
        box_b: (xmin, ymin, xmax, ymax)
    Returns:
        intersection_area: intersection area between two points
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    return intersection_area


def _iou_(bbox_a, bbox_b):
    """Intersection over union"""
    # compute the area of both the prediction and ground truth rectangles
    box_A_area = abs((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    box_B_area = abs((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

    inter_area = get_inter_area(bbox_a, bbox_b)
    union = float(box_A_area + box_B_area - inter_area)

    return inter_area / union


def _overlap_(test_poly, true_polys, rtree_index):
    """Calculate overlap between one polygon and all ground truth by area
    Args:
        test_poly: pd.Series
        true_polys: pd.DataFrame
        rtree_index: index of bounding boxes
    Returns:
        results: pd.DataFrame
    """
    prediction_id = []
    truth_id = []
    area = []
    matched_list = list(rtree_index.intersection(test_poly['bbox']))
    for index, true_row in true_polys.iterrows():
        if index in matched_list:
            intersection_area = get_inter_area(true_row['bbox'], test_poly['bbox'])
        else:
            intersection_area = 0

        prediction_id.append(test_poly['prediction_id'])
        truth_id.append(true_row['truth_id'])
        area.append(intersection_area)

    results = pd.DataFrame({
        'prediction_id': prediction_id,
        'truth_id': truth_id,
        'area': area
    })

    return results


def _overlap_all_(pred_polys, true_polys, rtree_index):
    """Find area of overlap among all sets of ground truth and prediction"""
    results = []
    for i, row in pred_polys.iterrows():
        result = _overlap_(test_poly=row,
                           true_polys=true_polys,
                           rtree_index=rtree_index)
        results.append(result)

    results = pd.concat(results, ignore_index=True)

    return results


def calculate_iou(predictions, ground_truth):
    """Intersection over union"""
    pred_polys = predictions.copy()
    true_polys = ground_truth.copy()

    true_polys['bbox'] = true_polys.apply(lambda x: (x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    pred_polys['bbox'] = pred_polys.apply(lambda x: (x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    # Create index columns
    true_polys['truth_id'] = true_polys.index.values
    pred_polys['prediction_id'] = pred_polys.index.values

    rtree_index = create_rtree_from_polygon(true_polys['bbox'])
    # find overlap among all sets
    overlap_df = _overlap_all_(pred_polys, true_polys, rtree_index)

    # Create cost matrix for assignment
    matrix = overlap_df.pivot(index='truth_id', columns='prediction_id', values='area').values
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)

    # Create IoU dataframe, match those predictions and ground truth, IoU = 0 for all others, they will get filtered out
    iou_df = []
    for index, row in true_polys.iterrows():
        y_true = true_polys.loc[index]
        if index in row_ind:
            matched_id = col_ind[np.where(index == row_ind)[0][0]]
            y_pred = pred_polys[pred_polys['prediction_id'] == matched_id].loc[matched_id]
            iou = _iou_(y_pred['bbox'], y_true['bbox'])
            pred_bbox = y_pred['bbox']
        else:
            iou = 0
            matched_id = None
            pred_bbox = None

        iou_df.append(pd.DataFrame({
            'prediction_id': [matched_id],
            'truth_id': [index],
            'IoU': iou,
            'prediction_bbox': [pred_bbox],
            'truth_bbox': [y_true['bbox']]
        }))

    iou_df = pd.concat(iou_df)
    iou_df = iou_df.merge(true_polys['truth_id'])

    return iou_df


def find_false_positives(predictions, ground_truth, threshold=0.5):
    iou_results = calculate_iou(predictions, ground_truth)
    iou_results = iou_results.dropna()  # drop the rows for labels with no prediction
    filt = iou_results['IoU'] <= threshold

    fp_iou = iou_results[filt]
    fp_predictions_df = predictions.iloc[fp_iou['prediction_id']]

    return fp_predictions_df
