import os, sys
import argparse

import numpy as np
import torch
import re
import time
import pickle
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-path', type=str, default='/disk0/bjgbiesseck/face_recognition/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES')
    parser.add_argument('--input-ext', type=str, default='_id_feat.pt')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--num-classes', type=int, default=-1)   # -1 == all classes
    args = parser.parse_args()
    return args


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_paths_with_dir_classes(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    if isinstance(file_extension, str): file_extension = [file_extension]
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



if __name__ == '__main__':
    args = parse_args()

    args.input_path = args.input_path.rstrip('/')
    assert os.path.isdir(args.input_path), f"Error, no such dir \'{args.input_path}\'"
    if not args.output_path:
        args.output_path = args.input_path + '_DIMENS_REDUCTION'
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


    if not 'embedds_feats_2d_tsne' in dict_dataset_all_embedds:
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        print(f'Reducing dimensionality of embeddings...')
        embedds_feats_2d_tsne = tsne.fit_transform(embedds_feats)
        dict_dataset_all_embedds['embedds_feats_2d_tsne'] = embedds_feats_2d_tsne
        print(f'Saving embedds_feats_2d_tsne: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading embedds_feats_2d_tsne: \'{dataset_all_embedds_file_path}\'')
        embedds_feats_2d_tsne = dict_dataset_all_embedds['embedds_feats_2d_tsne']
    print('embedds_feats_2d_tsne.shape:', embedds_feats_2d_tsne.shape)
    print(f'------------------')


    if args.num_classes > -1 and args.num_classes < max(embedds_classes_int):
        idx_last_sample_target_class = get_samples_indexes_corresponding_classes(embedds_classes_int, args.num_classes)
        embedds_feats_2d_tsne = embedds_feats_2d_tsne[:idx_last_sample_target_class,:]
        embedds_classes_int   = embedds_classes_int[:idx_last_sample_target_class]
        # print('embedds_classes_int:', embedds_classes_int)
    

    num_classes_to_plot = args.num_classes if args.num_classes > -1 else max(embedds_classes_int)+1
    title_scatter_plot_embeddings_2d = f"{args.input_path.split('/')[-2]} - Face Embeddings (t-SNE) - num-classes: {num_classes_to_plot}"
    file_name_scatter_plot_embeddings_2d = f'scatter_plot_embeddings_2d_num-classes={num_classes_to_plot}.png'
    path_scatter_plot_embeddings_2d = os.path.join(args.output_path, file_name_scatter_plot_embeddings_2d)
    print(f"Saving scatter plot of embeddings 2D: \'{path_scatter_plot_embeddings_2d}\'")
    print(f"    Plotting samples: {embedds_feats_2d_tsne.shape[0]}")
    save_scatter_plot_embeddings_2d(embedds_feats_2d_tsne, embedds_classes_int, title_scatter_plot_embeddings_2d, path_scatter_plot_embeddings_2d)
    print(f'------------------')


    print('\nFinished!')