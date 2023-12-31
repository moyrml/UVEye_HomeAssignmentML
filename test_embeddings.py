import pandas as pd
import pickle as pkl
from pathlib import Path

from utils.plot_utils import plot_2d_embeddings_scatter
from train_embeddings import reduce_dim


def test_embeddings(embeddings, labels, cluster_algo, dim_reduction_algo=None, master_dir=None):
    draw_output = False
    if dim_reduction_algo is not None:
        embeddings, pca = reduce_dim(
            embeddings,
            pca=dim_reduction_algo
        )

        if pca.n_components == 2 and master_dir is not None:
            draw_output = True

    cluster_affiliation = cluster_algo.predict(embeddings)
    embeddings['cluster affiliation'] = cluster_affiliation
    if hasattr(cluster_algo, 'class_size_prior') and cluster_algo.class_size_prior is not None:
        embeddings['cluster affiliation'] = embeddings['cluster affiliation'].replace(cluster_algo.class_size_prior)

    if draw_output:
        embeddings['label'] = labels
        embeddings['label'] = embeddings['label'].astype(str)
        if master_dir is not None:
            fig = plot_2d_embeddings_scatter(
                embeddings, embedding_model['cluster'], label_col='cluster affiliation'
            )
            fig.write_html(master_dir / 'latent_vectors' / set_type / '2d_pca_embeddings.html')

    return embeddings


if __name__ == '__main__':
    master_dir = 'outputs/December_30_2023_10_59AM/'
    set_type = 'test'

    master_dir = Path(master_dir)
    test_embeddings_data_file = master_dir / 'latent_vectors' / set_type / 'names_labels_embeddings.pkl'
    assert test_embeddings_data_file.exists(), f'Cannot find test embeddings file at {test_embeddings_data_file}'
    with open(test_embeddings_data_file, 'rb') as f:
        test_embeddings_data = pkl.load(f)

    embedding_model_file = master_dir / 'embedding_model.pkl'
    assert embedding_model_file.exists(), f'Cannot find embedding model at {embedding_model_file}'
    with open(embedding_model_file, 'rb') as f:
        embedding_model = pkl.load(f)

    embeddings = pd.DataFrame(test_embeddings_data['embeddings'])
    embeddings = test_embeddings(
        embeddings,
        test_embeddings_data['labels'],
        embedding_model['cluster'],
        embedding_model['reduce_dim'],
        master_dir=master_dir
    )

    print('Done')
