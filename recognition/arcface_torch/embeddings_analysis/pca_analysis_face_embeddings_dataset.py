import os, sys
import argparse

import numpy as np
import torch
import re
import time
import pickle
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-path', type=str, default='/hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS')
    parser.add_argument('--input-ext', type=str, default='_id_feat.pt')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--num-classes', type=int, default=-1)   # -1 == all classes
    parser.add_argument('--dataset-name', type=str, default='CASIA-WebFace')
    args = parser.parse_args()
    return args


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_paths_with_dir_classes(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    print(f'Found files: {len(file_list)}', end='\r')
    print('\nSorting paths...')
    file_list = natural_sort(file_list)
    classes_str_list = [f.split('/')[-2] for f in file_list]
    classes_str_unique_list = natural_sort(list(set(classes_str_list)))
    classes_str_unique_int  = list(range(0, len(classes_str_unique_list)))
    dict_classes_str_int = {class_str:class_int for (class_str,class_int) in zip(classes_str_unique_list,classes_str_unique_int)}
    # print('dict_classes_str_int:', dict_classes_str_int)
    # sys.exit(0)
    # for i, class_str in enumerate():
    classes_int_list = [dict_classes_str_int[class_str] for class_str in classes_str_list]

    return file_list, classes_str_list, classes_int_list


def load_embedding(embedd_path=''):
    if embedd_path.endswith('.pt'):
        embedd = torch.load(embedd_path)
    elif embedd_path.endswith('.npy'):
        embedd = np.load(embedd_path)
    else:
        raise Exception(f'File format not supported: \'{embedd_path}\'')
    return embedd


def save_dict_to_pickle(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_samples_indexes_corresponding_classes(embedds_classes_int, num_classes):
    last_sample_target_class = -1
    for idx_sample_class, sample_class in enumerate(embedds_classes_int):
        if sample_class == num_classes:
            last_sample_target_class = idx_sample_class
            break
    return last_sample_target_class


def save_scatter_plot_embeddings_2d(embedds_2d, embedds_classes_int, title_scatter_plot, path_scatter_plot):
    directory = os.path.dirname(path_scatter_plot)
    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(10, 8), dpi=150)
    scatter = plt.scatter(
        embedds_2d[:, 0], 
        embedds_2d[:, 1], 
        c=embedds_classes_int, 
        cmap='jet', 
        alpha=0.6,
        edgecolors='none',
        # s=30 # Size of dots
        s=10 # Size of dots
    )

    plt.title(title_scatter_plot, fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    cbar = plt.colorbar(scatter, label='Class ID')
    cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x):05d}")
    cbar.update_ticks()

    plt.savefig(path_scatter_plot, bbox_inches='tight')
    plt.close()



def save_plot_pca_explainability(pca, title='', path_chart='pca_explainability'):
    # Calculate variance ratios
    explained_variance_normalized = pca.explained_variance_ / np.sum(pca.explained_variance_)
    pca_cumulative_sum = np.cumsum(explained_variance_normalized)
    
    # Identify threshold points
    num_components_95variability = np.where(pca_cumulative_sum >= 0.95)[0][0] + 1
    num_components_99variability = np.where(pca_cumulative_sum >= 0.99)[0][0] + 1

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    n_features = len(explained_variance_normalized)
    x_range = np.arange(1, n_features + 1)

    # 1. Bar Chart: Individual Variance
    ax.bar(x_range, explained_variance_normalized, alpha=0.8, color='darkblue', 
           label='Individual Explained Variance')

    # 2. Step Plot: Cumulative Sum
    ax.step(x_range, pca_cumulative_sum, where='mid', color='skyblue', linewidth=1.5,
            label='Cumulative Explained Variance')

    # 3. Threshold Highlights
    thresholds = [
        (num_components_95variability, 0.95, 'red', '95%'),
        (num_components_99variability, 0.99, 'green', '99%')
    ]

    for n_comp, val, color, label in thresholds:
        ax.axvline(x=n_comp, color=color, linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=val, color=color, linestyle=':', alpha=0.5)
        # Dynamic positioning for text to avoid overlapping the top of the chart
        ax.text(n_comp + (n_features * 0.01), val - 0.07, f'{label} at {n_comp}', 
                color=color, fontweight='bold', fontsize=9, rotation=0)

    # 4. Format X-Axis (Rotation and Density)
    ax.set_xticks(x_range)
    # Only show a subset of labels if n_features is large (e.g., 512)
    if n_features > 20:
        # Show every 10th label if small, or more if very large
        interval = max(1, n_features // 20) 
        ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    
    # Rotate labels 90 degrees and reduce font size
    plt.xticks(rotation=90, fontsize=7)
    
    # 5. General Formatting
    ax.set_title(title if title else 'PCA Explained Variance Analysis', fontsize=14)
    ax.set_xlabel('Number of Principal Components', fontsize=11)
    ax.set_ylabel('Explainability Percentage (Variance Ratio)', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-5, n_features + 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # Save outputs
    plt.tight_layout()
    plt.savefig(path_chart, facecolor='white')
    plt.savefig(path_chart.replace('.png','.svg').replace('.jpg','.svg'))
    plt.close()



if __name__ == '__main__':
    args = parse_args()

    args.input_path = args.input_path.rstrip('/')
    assert os.path.isdir(args.input_path), f"Error, no such dir \'{args.input_path}\'"
    if not args.output_path:
        args.output_path = args.input_path + '_ANALYSIS_PCA'
    os.makedirs(args.output_path, exist_ok=True)

    if not type(args.input_ext) is list:
        args.input_ext = [args.input_ext]


    dataset_all_embedds_file_name = f"{args.input_path.split('/')[-1]}.pkl"
    dataset_all_embedds_file_path = os.path.join(args.output_path, dataset_all_embedds_file_name)
    if not os.path.isfile(dataset_all_embedds_file_path):
        print(f"Searching embeddings in \'{args.input_path}\'")
        embedds_paths, embedds_classes_str, embedds_classes_int = get_all_files_paths_with_dir_classes(args.input_path, file_extension=args.input_ext)
        # print('embedds_classes_str:', embedds_classes_str)
        # print('embedds_classes_int:', embedds_classes_int)
        # sys.exit(0)

        dict_dataset_all_embedds = {}
        dict_dataset_all_embedds['embedds_paths']       = embedds_paths
        dict_dataset_all_embedds['embedds_classes_str'] = embedds_classes_str
        dict_dataset_all_embedds['embedds_classes_int'] = embedds_classes_int

        print(f'Saving all embeddings paths: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading all embeddings paths: \'{dataset_all_embedds_file_path}\'')
        dict_dataset_all_embedds = load_dict_from_pickle(dataset_all_embedds_file_path)
        embedds_paths       = dict_dataset_all_embedds['embedds_paths']
        embedds_classes_str = dict_dataset_all_embedds['embedds_classes_str']
        embedds_classes_int = dict_dataset_all_embedds['embedds_classes_int']
    print('len(embedds_paths):', len(embedds_paths))
    print(f'------------------')


    if not 'embedds_feats' in dict_dataset_all_embedds:
        one_embedd = load_embedding(embedds_paths[0])
        print('one_embedd.shape:', one_embedd.shape)
        shape_embedds_feats = (len(embedds_paths), one_embedd.shape[1])
        print('Allocating matrix:', shape_embedds_feats)
        embedds_feats = np.zeros(shape_embedds_feats, dtype=float)
        print('Done!')

        for idx_embedd, path_embedd in enumerate(embedds_paths):
            start_time = time.time()
            print(f'{idx_embedd}/{len(embedds_paths)} - Loading embedding \'{path_embedd}\'', end='\r')
            one_embedd = load_embedding(path_embedd)
            one_embedd = one_embedd.cpu().detach().numpy()
            embedds_feats[idx_embedd] = one_embedd
        print()
        dict_dataset_all_embedds['embedds_feats'] = embedds_feats
        print(f'Saving all embeddings features: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading all embeddings features: \'{dataset_all_embedds_file_path}\'')
        embedds_feats = dict_dataset_all_embedds['embedds_feats']
    print('embedds_feats.shape:', embedds_feats.shape)
    print(f'------------------')


    if not 'pca' in dict_dataset_all_embedds:
        print(f'Computing Principal Components (PCA)')
        pca = PCA(n_components=embedds_feats.shape[1])
        pca.fit(embedds_feats)
        dict_dataset_all_embedds['pca'] = pca
        print(f'Saving pca: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading Principal Components (PCA): \'{dataset_all_embedds_file_path}\'')
        pca = dict_dataset_all_embedds['pca']
    print('pca:', pca)
    # print('dir(pca):', dir(pca))
    pc_vectors = pca.components_ * pca.explained_variance_[:, np.newaxis]
    explained_variance_normalized = pca.explained_variance_ / np.sum(pca.explained_variance_)
    pca_cumulative_sum = np.cumsum(explained_variance_normalized)
    num_components_95variability = np.where(pca_cumulative_sum >= 0.95)[0][0] + 1
    num_components_99variability = np.where(pca_cumulative_sum >= 0.99)[0][0] + 1
    print(f"95% variance reached at: {num_components_95variability} components")
    print(f"99% variance reached at: {num_components_99variability} components")
    print(f'------------------')


    title = f"{args.dataset_name} - PCA"
    path_pca_chart = os.path.join(args.output_path, 'pca_explainability.png')
    print(f'Saving PCA explainability chart: \'{path_pca_chart}\'')
    save_plot_pca_explainability(pca, title, path_pca_chart)
    print(f'------------------')


    print('\nFinished!')