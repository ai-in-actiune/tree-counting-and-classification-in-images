import matplotlib.pyplot as plt
import cv2 as cv


def plot_bboxes(image, df, color=(0, 255, 0), thickness=1):
    """
    Args:
        image: numpy.ndarray
        df: pd.DataFrame that contains coordinates xmin, ymin, xmax, ymax
        color: color fot the bb edges
        thickness: thickness of the bb edge
    Returns:
        void: plots the bounding boxes
    """
    for _, row in df.iterrows():
        p1 = (int(row['xmin']), int(row['ymin']))
        p2 = (int(row['xmax']), int(row['ymax']))
        cv.rectangle(image, p1, p2, color=color, thickness=thickness, lineType=cv.LINE_AA)

    plt.imshow(image[:, :, ::-1])
    plt.show()
