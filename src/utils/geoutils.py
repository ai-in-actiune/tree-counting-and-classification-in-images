import os
import pathlib
from typing import Tuple, Union, List


class ArgFormatException(Exception):
    """Raise if arguments are provided in a wrong format."""

    def __init__(self):
        super(ArgFormatException, self).__init__("Pixels should be provided in a list of ints or pairs of ints!")


class PixelOutOfBoundsException(Exception):
    """Raise if pixel is out of image bounds."""

    def __init__(self, pixel: Tuple[int, int], image: Tuple[int, int]):
        super(PixelOutOfBoundsException, self).__init__(f"Pixel is outside image boundaries.\nPixel: {pixel}"
                                                        f"\nImage: {image}")


def pixel_to_geo_coordinates(
        pixel_coordinates: Tuple[int, int],
        input_image_dims: Tuple[int, int],
        input_geo_coordinates: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    x, y = pixel_coordinates
    image_w, image_h = input_image_dims
    top_lat, top_long, bottom_lat, bottom_long = input_geo_coordinates
    if image_w < x or x < 0 or image_h < y or y < 0:
        raise PixelOutOfBoundsException(pixel_coordinates, input_image_dims)
    if top_lat > bottom_lat:
        top_lat, bottom_lat = bottom_lat, top_lat
    if top_long > bottom_long:
        top_long, bottom_long = bottom_long, top_long
    return top_lat + x / image_w * (bottom_lat - top_lat), top_long + x / image_w * (bottom_long - top_long)


def get_coordinates_from_image(image_path_or_fp: Union[str, os.PathLike], *pixels: Union[int, Tuple[int, int]]) -> \
        List[Tuple[float, float]]:
    """
    Transform the provided pixel coordinates into geocoordinates.
    >>> get_coordinates_from_image(
    >>>     'image.tiff',
    >>>     10, 20,
    >>>     (20, 30)
    >>> )

    Args:
        pixels: A set of pixel coordinates.
        image_path_or_fp: A path to a GeoTIFF image

    Returns:
        The geolocation of the provided pixel coordinates
    """
    import rasterio
    if isinstance(image_path_or_fp, str):
        image_path_or_fp = pathlib.Path(image_path_or_fp)
    if not image_path_or_fp.exists():
        raise FileNotFoundError("The provided path does not exist!")
    tiff_image: rasterio.DatasetReader = rasterio.open(image_path_or_fp)
    left, bottom, right, top = tiff_image.bounds
    width, height = tiff_image.shape
    answers = []
    stack = []
    if not pixels:
        raise ArgFormatException
    for pixel in pixels:
        if isinstance(pixel, tuple):
            if stack:
                raise ArgFormatException
            if len(pixel) != 2:
                raise ArgFormatException
            stack = [*pixel]
        elif isinstance(pixel, int):
            stack.append(pixel)
        if len(stack) == 2:
            answers.append(pixel_to_geo_coordinates((stack[0], stack[1]), (width, height), (top, left, bottom, right)))
            stack.clear()
    if stack:
        raise ArgFormatException
    return answers

