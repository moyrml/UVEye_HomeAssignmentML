import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from utils.model import AE
from utils.dataset import ImageDataset


def test_ae(model_path, test_dataloader, device, output_dir):
    """
    Visually test AE quality. This is an exhaustive process of image creation. We don't need to create ALL images since
    There is high dependency them.

    :param model_path: str. Path to weights of AE model
    :param test_dataloader:
    :param device:
    :param output_dir: pathlib.Path object
    :return:
    """
    model = AE(dict(
        depth=6,
        expand_factor=2,
        latent_dim=32,
        activation_func=nn.GELU,
        input_dim=512
    ))
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    with torch.no_grad():
        for batch_i, (images, label) in enumerate(test_dataloader):
            recon_batch = model(images)
            recon_batch += 1
            recon_batch *= 125
            for batch_item in range(recon_batch.shape[0]):
                recon_image = recon_batch[batch_item].permute([1, 2, 0])
                recon_image = np.array(recon_image).astype('uint8')

                cv2.imwrite(str(output_dir / f'{batch_i}_{batch_item}.jpg'), recon_image)


if __name__ == '__main__':
    model_path = '/home/moshe/UVEye_HomeAssignmentML/outputs/December_29_2023_09_59PM/ae_model.pth'
    output_dir = '/home/moshe/UVEye_HomeAssignmentML/outputs/December_29_2023_09_59PM/ae_testing'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_dataset = ImageDataset(
        'data/black_white_dataset',
        path_label_loc=3,
        dataset_name='train',
        scale_images_to=512,
        normalize=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        batch_size=1
    )

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_ae(model_path, train_dataloader, device, output_dir)

