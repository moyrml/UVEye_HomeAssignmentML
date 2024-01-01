import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser
import json

from utils.model import AE
from utils.dataset import ImageDataset
from utils.activation_func_mapping import get_activation_func_from_name

def test_ae(model_path, test_dataloader, device, output_dir, train_config):
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
        depth=train_config['ae_depth'],
        expand_factor=train_config['ae_expand_factor'],
        latent_dim=train_config['ae_latent_dim'],
        activation_func=get_activation_func_from_name(train_config['ae_activation_func']),
        input_dim=train_config['image_scale']
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
    parser = ArgumentParser()
    parser.add_argument('--model_path', default=None)
    args = parser.parse_args()

    if args.model_path is None:
        last_run_file = Path('train_ae_last_output_folder.json')
        assert last_run_file.exists(), f'No model_path specified and cannot find train_ae_last_output_folder.json'

        with open(last_run_file, 'r') as f:
            model_path = Path(json.load(f)['output_dir'])
    else:
        model_path = Path(args.model_path)
    print(f'Using model directory {model_path}')

    model_path = Path(args.model_path) / 'ae_model.pth'
    output_dir = model_path.parent / 'ae_testing'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_config_path = model_path.parent / 'run_config.json'
    with open(train_config_path) as f:
        train_config = json.load(f)

    train_dataset = ImageDataset(
        'data/categories_dataset',
        dataset_name='test',
        scale_images_to=train_config['image_scale'],
        normalize=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        batch_size=1
    )

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_ae(model_path, train_dataloader, device, output_dir, train_config)
