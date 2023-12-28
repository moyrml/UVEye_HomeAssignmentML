import pandas as pd
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
from functools import partial
import plotly_express as px
import plotly.graph_objs as go
import cv2


def gather_images(data_dir):
    """
    Find all image files in data_dir. Assumes all images are png files!

    :param str data_dir: Path to master directory
    :return: list. List of `pathlib.Path` object pointing to images.
    """

    images = list(Path(data_dir).rglob('*.png'))
    return images


def get_single_image_size_and_set(image_path, set_location):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    channel_mean = [image[c].mean() for c in range(image.shape[2])]
    width, height = image.size

    image_set = image_path.parts[set_location]
    return image_set, width, height, *channel_mean


def get_image_data(images, set_location=2):
    """
    Get all image sizes and set (train|test) and create scatter plot.
    Assumes the set is in the path and at a given location in the path

    :param images: list of `pathlib.Path` objects.
    :return: `pandas.DataFrame`.
    """
    part = partial(get_single_image_size_and_set, set_location=set_location)
    with Pool(1) as pool:
        size_set = pool.map(part, images)
    size_set = pd.DataFrame(size_set, columns=['set', 'width', 'height'])

    return size_set


def plot_width_height_scatter(image_data_df, output_path=None):
    """
    Create height x width plot by set.

    :param image_data_df:
    :param output_path: pathlib.Path to folder to which the plot will be saved. Empty for no saving.
    :return: `plotly.graph_objs.Figure`
    """
    max_value = image_data_df[['width', 'height']].max().max()
    min_value = image_data_df[['width', 'height']].min().min()
    x_eq_y = pd.DataFrame([[min_value, max_value], [min_value, max_value]], index=['x', 'y']).T

    fig = px.scatter(image_data_df, x='width', y='height', color='set')
    group_size = image_data_df.groupby('set').size().to_dict()
    for trace in fig.data:
        trace['name'] = f"{trace['name']} - N = {group_size[trace['name']]}"

    diag = px.line(x_eq_y, x='x', y='y')

    combined = go.Figure(data=fig.data + diag.data)
    combined.layout.title = "Image height x width distribution"

    if output_path is not None:
        combined.write_html(output_path / 'width_height_set.html')

    return combined


if __name__ == '__main__':
    data_dir = 'data/black_white_dataset'
    output_path = 'outputs/eda/'

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    images = gather_images(data_dir)

    image_data_df = get_image_data(images)
    plot_width_height_scatter(image_data_df, output_path=output_path)
