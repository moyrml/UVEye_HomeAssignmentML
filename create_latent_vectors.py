from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import pickle as pkl

import torch.optim
from torch.utils.data import DataLoader

from utils.model import AE
from utils.dataset import ImageDataset
from utils.activation_func_mapping import get_activation_func_from_name


def create_latent_vectors(ae_model, dataloader, device, output_dir):
    embeddings = []
    image_names = []
    labels = []

    ae_model.eval()
    with torch.no_grad():
        for batch_i, (images, label, image_name) in enumerate(dataloader):
            batch_embeddings = ae_model.encoder(images.to(device))

            embeddings.append(batch_embeddings)
            image_names.extend(image_name)
            labels.extend([l.item() for l in label])

    embeddings = np.concatenate(embeddings)

    with open(output_dir / 'names_labels_embeddings.pkl', 'wb') as f:
        pkl.dump(
            dict(
                embeddings=embeddings,
                image_names=image_names,
                labels=labels,
                label_encoder=dataloader.dataset.label_encoder
            ),
            f
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--data_location', default='data/black_white_dataset')
    # Im doing this override because im training on colab and sometimes testing on my pc
    parser.add_argument('--data_label_loc', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--set_type', default='train')
    args = parser.parse_args()

    if args.model_path is None:
        last_run_file = Path('train_ae_last_output_folder.json')
        assert last_run_file.exists(), f'No model_path specified and cannot find train_ae_last_output_folder.json'

        with open(last_run_file, 'r') as f:
            model_path = Path(json.load(f)['output_dir'])
    else:
        model_path = Path(args.model_path)

    output_dir = model_path / 'latent_vectors' / args.set_type
    output_dir.mkdir(exist_ok=True, parents=True)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_config_path = model_path / 'run_config.json'
    with open(train_config_path) as f:
        train_config = json.load(f)
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=True)

    dataset = ImageDataset(
        args.data_location,
        path_label_loc=args.data_label_loc if args.data_label_loc != -1 else train_config['data_label_loc'],
        dataset_name=args.set_type,
        scale_images_to=train_config['image_scale'],
        normalize=True,
        return_filename=True
    )

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers if args.num_workers != -1 else train_config['num_workers'],
        shuffle=True,
        batch_size=args.bs if args.bs != -1 else train_config['bs'],
    )

    model = AE(dict(
        depth=train_config['ae_depth'],
        expand_factor=train_config['ae_expand_factor'],
        latent_dim=train_config['ae_latent_dim'],
        activation_func=get_activation_func_from_name(train_config['ae_activation_func']),
        input_dim=train_config['image_scale']
    )).to(device)
    model.load_state_dict(torch.load(model_path / 'ae_model.pth', map_location=device))

    create_latent_vectors(model, dataloader, device, output_dir)
