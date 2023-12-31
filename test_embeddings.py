import pandas as pd
import pickle as pkl
import json
from pathlib import Path
from argparse import ArgumentParser
from utils.plot_utils import plot_2d_embeddings_scatter
from train_embeddings import reduce_dim


def test_embeddings(embeddings, labels, cluster_algo, dim_reduction_algo=None, output_dir=None):
    draw_output = False
    if dim_reduction_algo is not None:
        embeddings, pca = reduce_dim(
            embeddings,
            pca=dim_reduction_algo
        )

        if pca.n_components == 2 and output_dir is not None:
            draw_output = True

    cluster_affiliation = cluster_algo.predict(embeddings)
    embeddings['cluster affiliation'] = cluster_affiliation
    if hasattr(cluster_algo, 'class_size_prior') and cluster_algo.class_size_prior is not None:
        embeddings['cluster affiliation'] = embeddings['cluster affiliation'].replace(cluster_algo.class_size_prior)

    if draw_output:
        embeddings['label'] = labels
        embeddings['label'] = embeddings['label'].astype(str)
        if output_dir is not None:
            fig = plot_2d_embeddings_scatter(
                embeddings, embedding_model['cluster'], label_col='cluster affiliation'
            )
            fig.write_html(output_dir / '2d_pca_embeddings.html')

            fig = plot_2d_embeddings_scatter(
                embeddings, embedding_model['cluster'], label_col='label'
            )
            fig.write_html(output_dir / 'ground_truth_2d_pca_embeddings.html')

    return embeddings


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument(
        '--embeddings_file',
        default=None,
        help='path to embeddings. Do not specify if you want to use the default of '
             'model_dir/latent_vectors/$set_type/names_labels_embeddings.pkl'
    )
    parser.add_argument('--set_type', default='test')
    args = parser.parse_args()

    if args.model_path is None:
        last_run_file = Path('train_ae_last_output_folder.json')
        assert last_run_file.exists(), f'No model_path specified and cannot find train_ae_last_output_folder.json'

        with open(last_run_file, 'r') as f:
            model_path = Path(json.load(f)['output_dir'])
    else:
        model_path = Path(args.model_path)

    model_path = Path(model_path)

    if args.embeddings_file is None:
        test_embeddings_data_file = model_path / 'latent_vectors' / args.set_type / 'names_labels_embeddings.pkl'
        assert test_embeddings_data_file.exists(), f'Cannot find test embeddings file at {test_embeddings_data_file}'
    else:
        test_embeddings_data_file = args.embeddings_file
    with open(test_embeddings_data_file, 'rb') as f:
        test_embeddings_data = pkl.load(f)

    embedding_model_file = model_path / 'embedding_model.pkl'
    assert embedding_model_file.exists(), f'Cannot find embedding model at {embedding_model_file}'
    with open(embedding_model_file, 'rb') as f:
        embedding_model = pkl.load(f)

    test_embeddings_data['labels'] = [
        test_embeddings_data['label_encoder'].decode(l) for l in test_embeddings_data['labels']
    ]

    embeddings = pd.DataFrame(test_embeddings_data['embeddings'])
    embeddings = test_embeddings(
        embeddings,
        test_embeddings_data['labels'],
        embedding_model['cluster'],
        embedding_model['reduce_dim'],
        output_dir=model_path / 'latent_vectors' / args.set_type
    )

    embeddings['image name'] = test_embeddings_data['image_names']
    embeddings[['image name', 'label', 'cluster affiliation']].to_csv(model_path / 'prediction.csv', index=False)

    print('Done')
