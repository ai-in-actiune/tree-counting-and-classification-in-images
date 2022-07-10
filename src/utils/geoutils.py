from typing import Tuple


class PixelOutOfBoundsException(Exception):
    def __init__(self):
        super(PixelOutOfBoundsException, self).__init__("Pixel is outside image boundaries.")


def pixel_to_geo_coordinates(
        pixel_coordinates: Tuple[int, int],
        input_image_dims: Tuple[int, int],
        input_geo_coordinates: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    x, y = pixel_coordinates
    image_w, image_h = input_image_dims
    top_lat, top_long, bottom_lat, bottom_long = input_geo_coordinates
    if image_w < x < 0 or image_h < y < 0:
        raise PixelOutOfBoundsException()
    if top_lat > bottom_lat:
        top_lat, bottom_lat = bottom_lat, top_lat
    if top_long > bottom_long:
        top_long, bottom_long = bottom_long, top_long
    return top_lat + x / image_w * (bottom_lat - top_lat), top_long + x / image_w * (bottom_long - top_long)


if __name__ == '__main__':
    print(pixel_to_geo_coordinates((12, 13), (20, 20), (48.17, 27.19, 50.12, 28.22)))
