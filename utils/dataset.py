from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from load_image import load_image
from label_encoder import LabelEncoder


class ImageDataset(Dataset):
    def __init__(
            self,
            data_master_dir,
            path_background_loc=None,
            path_label_loc=None,
            scale_images_to=None,
            dataset_name='train'
    ):
        """
        Dataset object.
        Passed to `torch.utils.data.DataLoader`,  and when the latter is called it uses a multi-threaded manner
        to call the dataset and pull examples to form batches.

        :param str data_master_dir: str. Path to directory containing images, e.g. `data/black_white_dataset`
        :param path_background_loc: int or None. Location in path denoting the background color. Relevant only to
            train time, hence should be None for testing time.
        :param path_label_loc: int or None. Location in path denoting the label location. Relevant only for testing
            time, hence should be None for training time.
        :param int scale_images_to: Size target for images. Leave None for no scaling, but for batching purposes the
            images should be of equal sizes. EDA tells of square images so the resize is to size
            (scale_images_to,scale_images_to).
        :param dataset_name: str. dataset_name=train|test, other values will produce a ValueError.
        """
        images = list(Path(data_master_dir).rglob('*.png'))
        image_names = [p.stem for p in images]
        background_color = [np.nan] * len(images)
        label = [np.nan] * len(images)

        if dataset_name not in ['train', 'test']:
            raise ValueError(f"Incorrect dataset name: {dataset_name}")

        if dataset_name == 'train':
            background_color = [p.parts[path_background_loc] for p in images]
            label_encoder = LabelEncoder(['black', 'white'])
        else:
            label = [p.parts[path_label_loc] for p in images]
            label_encoder = LabelEncoder(set(label))

        images = [str(p) for p in images]
        df = pd.DataFrame(
            [image_names, images, background_color, label],
            index=['image name', 'image path', 'background color', 'label']
        ).T

        self.df = df
        self.dataset_name = dataset_name
        self.scale_images_to = scale_images_to
        self.label_encoder = label_encoder

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.loc[item]

        image = load_image(row['image path'], self.scale_images_to)
        image = torch.Tensor(image)
        image = image.permute([2, 1, 0])

        if self.dataset_name == 'test':
            label = self.label_encoder.encode(row.label)
        else:
            label = self.label_encoder.encode(row['background color'])

        return image, label


if __name__ == '__main__':
    image_loader = ImageDataset(
        '../data/black_white_dataset',
        path_background_loc=4,
        dataset_name='train',
        scale_images_to=500
    )

    image, background_color = image_loader[0]

    image_test_loader = ImageDataset(
        '../data/categories_dataset',
        path_label_loc=4,
        dataset_name='test',
        scale_images_to=500
    )

    test_image, test_label = image_test_loader[0]
    print('Done')
