from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.load_image import load_image
from utils.label_encoder import LabelEncoder


class ImageDataset(Dataset):
    def __init__(
            self,
            data_master_dir,
            scale_images_to=None,
            dataset_name='train',
            normalize=False,
            return_filename=False
    ):
        """
        Dataset object.
        Passed to `torch.utils.data.DataLoader`,  and when the latter is called it uses a multi-threaded manner
        to call the dataset and pull examples to form batches.

        :param str data_master_dir: str. Path to directory containing images, e.g. `data/black_white_dataset`
        :param int scale_images_to: Size target for images. Leave None for no scaling, but for batching purposes the
            images should be of equal sizes. EDA tells of square images so the resize is to size
            (scale_images_to,scale_images_to).
        :param dataset_name: str. dataset_name=train|test, other values will produce a ValueError.
        """
        images = list(Path(data_master_dir).rglob('*.png'))
        image_names = [p.stem for p in images]
        image_set = [p.parent.parent.stem for p in images]
        background_color = [np.nan] * len(images)
        label = [np.nan] * len(images)

        assert dataset_name in ['train', 'test'], f"Incorrect dataset name: {dataset_name}"

        property = [p.parent.stem for p in images]
        label_encoder = LabelEncoder(set(property))

        if dataset_name == 'train':
            background_color = property
        else:
            label = property

        images = [str(p) for p in images]
        df = pd.DataFrame(
            [image_set, image_names, images, background_color, label],
            index=['set', 'image name', 'image path', 'background color', 'label']
        ).T

        df = df.loc[df.set == dataset_name]
        df.reset_index(inplace=True)

        self.df = df
        self.dataset_name = dataset_name
        self.scale_images_to = scale_images_to
        self.label_encoder = label_encoder
        self.normalize = normalize
        self.return_filename = return_filename

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.loc[item]

        image = load_image(row['image path'], self.scale_images_to)
        image = torch.Tensor(image)
        image = image.permute([2, 0, 1])

        if self.normalize:
            # Normalize to [-1,1]
            image -= image.min()
            image /= image.max()

            image *= 2
            image -= 1

        if self.dataset_name == 'test':
            label = self.label_encoder.encode(row.label)
        else:
            label = self.label_encoder.encode(row['background color'])

        if self.return_filename:
            return image, label, row['image name']

        return image, label

    def describe(self):
        print(f'Loaded {self.dataset_name} dataset with {len(self)} samples')


if __name__ == '__main__':
    image_loader = ImageDataset(
        '../data/black_white_dataset',
        dataset_name='train',
        scale_images_to=500,
    )

    image, background_color = image_loader[0]

    image_test_loader = ImageDataset(
        '../data/categories_dataset',
        dataset_name='test',
        scale_images_to=500,
    )

    test_image, test_label = image_test_loader[0]
    print('Done')
