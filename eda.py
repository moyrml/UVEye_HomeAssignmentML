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
    width, height, _ = image.shape

    image_set = image_path.parts[set_location]
    return image_set, image_path.stem, image_path, width, height, *channel_mean


def get_image_data(images, set_location=2):
    """
    Get all image sizes and set (train|test) and create scatter plot.
    Assumes the set is in the path and at a given location in the path

    :param images: list of `pathlib.Path` objects.
    :return: `pandas.DataFrame`.
    """
    part = partial(get_single_image_size_and_set, set_location=set_location)
    with Pool() as pool:
        size_set = pool.map(part, images)
    size_set = pd.DataFrame(
        size_set,
        columns=['set', 'image name', 'image path', 'width', 'height', 'R mean', 'G mean', 'B mean']
    )

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


def plot_channel_means(image_data_df, background_color_loc=3, output_path=None):
    image_data_df['background color'] = image_data_df['image path'].apply(lambda x: Path(x).parts[background_color_loc])
    image_data_df.sort_values(['background color', 'set'], inplace=True)

    names = image_data_df.apply(lambda x: f"{x['image name']} - {x['set']} ({x['background color']})", axis='columns').tolist()

    fig = go.Figure()
    fig.add_traces([
        go.Scatter(y=image_data_df['R mean'], x=names, mode='markers', marker_color='red', name='R', showlegend=True),
        go.Scatter(y=image_data_df['G mean'], x=names, mode='markers', marker_color='green', name='G', showlegend=True),
        go.Scatter(y=image_data_df['B mean'], x=names, mode='markers', marker_color='blue', name='B', showlegend=True),
    ])

    black_names = [n for n in names if n.endswith('(black)')]
    fig.add_vrect(
        x0=black_names[0], x1=black_names[-1], fillcolor="gray", opacity=0.25,
        annotation_text="black background", annotation_position="top left"
    )
    fig.layout.title = 'Channel distribution per data point'

    if output_path is not None:
        fig.write_html(output_path / 'channel_distribution.html')


if __name__ == '__main__':
    data_dir = 'data/black_white_dataset'
    output_path = 'outputs/eda/'

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    images = gather_images(data_dir)

    image_data_df = get_image_data(images)
    plot_channel_means(image_data_df, output_path=output_path)
    plot_width_height_scatter(image_data_df, output_path=output_path)
