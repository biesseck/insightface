import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd
import copy
import re
import random
import glob


class Loader_HDA_Doppelganger:
    def __init__(self):
        pass


    def str_to_bool(self, s):
        true_values = {'yes', 'true', 't', 'on', '1'}
        false_values = {'no', 'false', 'f', 'off', '0'}
        
        s = s.strip().lower()
        if s in true_values:
            return True
        elif s in false_values:
            return False
        else:
            raise ValueError(f"Invalid truth value {s!r}")


    def load_protocol(self, protocol_file):
        with open(protocol_file, 'r') as file:
            all_lines = [line.strip() for line in file.readlines()]
            if all_lines[0].startswith('SAMPLE0,SAMPLE1,LABEL'):
                all_lines = all_lines[1:]
            protocol = len(all_lines) * [None]
            # print('all_lines:', all_lines)

            for i, line in enumerate(all_lines):
                # print('line:', line)
                line_data = line.split(',')
                # sys.exit(0)
                sample0      = line_data[0]
                sample1      = line_data[1]
                pair_label   = int(self.str_to_bool(line_data[2]))

                pair = {}
                pair['sample0']    = sample0
                pair['sample1']    = sample1
                pair['pair_label'] = pair_label

                protocol[i] = pair
            
            return protocol


    def get_all_files_in_path(self, folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
        def natural_sort(l):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        file_list = []
        for root, _, files in os.walk(folder_path):
            for filename in files:
                path_file = os.path.join(root, filename)
                for ext in file_extension:
                    if pattern in path_file and path_file.lower().endswith(ext.lower()):
                        file_list.append(path_file)
                        print(f'Found files: {len(file_list)}', end='\r')
        print()
        # file_list.sort()
        file_list = natural_sort(file_list)
        return file_list


    def update_paths(self, protocol, data_dir, data_dir_positive_samples, replace_ext='.png', inplace=True):
        if not inplace:
            protocol = copy.deepcopy(protocol)

        indexes_invalid_pairs = []
        num_img_multiple_faces = 0
        for i, pair in enumerate(protocol):
            sample0 = pair['sample0']
            sample1 = pair['sample1']
            pair_label = pair['pair_label']

            if replace_ext != '':
                curr_ext = sample0.split('.')[-1]
                sample0 = sample0.replace('.'+curr_ext, replace_ext)
                sample1 = sample1.replace('.'+curr_ext, replace_ext)
            
            sample0_name, sample0_ext = os.path.splitext(sample0)
            sample1_name, sample1_ext = os.path.splitext(sample1)

            if pair_label == 0:    # negative pairs from 'HDA-Doppelgaenger'
                sample0_pattern = os.path.join(data_dir, '/'.join(sample0_name.split('/')[1:])+'*'+sample0_ext).replace('[','*').replace(']','*')
                sample1_pattern = os.path.join(data_dir, '/'.join(sample1_name.split('/')[1:])+'*'+sample1_ext).replace('[','*').replace(']','*')
            else:    # positive pairs from 'FRGC'
                sample0_pattern = os.path.join(data_dir_positive_samples, '/'.join(sample0_name.split('/')[2:])+'*'+sample0_ext).replace('[','*').replace(']','*')
                sample1_pattern = os.path.join(data_dir_positive_samples, '/'.join(sample1_name.split('/')[2:])+'*'+sample1_ext).replace('[','*').replace(']','*')
            
            path_sample0 = self.natural_sort(glob.glob(sample0_pattern))
            assert len(path_sample0) > 0, f'Error, no file found with pattern \'{sample0_pattern}\''
            path_sample0 = path_sample0[0]

            path_sample1 = self.natural_sort(glob.glob(sample1_pattern))
            assert len(path_sample1) > 0, f'Error, no file found with pattern \'{sample1_pattern}\''
            path_sample1 = path_sample1[0]

            pair['sample0'] = path_sample0
            pair['sample1'] = path_sample1
        
        indexes_invalid_pairs = list(set(indexes_invalid_pairs))
        if len(indexes_invalid_pairs) > 0:
            protocol = self.remove_elements_by_indices(protocol, indexes_invalid_pairs)

        return protocol


    def load_images_protocol(self, data_dir=''):
        files_paths = self.get_all_files_in_path(data_dir)

        protocol = int(len(files_paths)/2) * [None]
        for i in range(0, len(files_paths), 2):
            sample0, sample1 = files_paths[i], files_paths[i+1]
            sample0_gender = sample0.split('/')[-2]
            sample1_gender = sample1.split('/')[-2]
            pair_label = 0

            pair = {}
            pair['sample0'] = sample0
            pair['sample0_gender'] = sample0_gender
            
            pair['sample1'] = sample1
            pair['sample1_gender'] = sample1_gender
            
            pair['pair_label'] = pair_label

            protocol[int(i/2)] = pair

        return protocol
    

    def natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    def load_dict_imgs_paths_by_subj(self, data_dir, ext=['.jpg', '.jpeg', '.png']):
        img_dict = {}
        num_found_files = 0
        for root, dirs, files in os.walk(data_dir):
            subfolder = os.path.basename(root)
            img_files = self.natural_sort([os.path.join(root, f) for f in files if any(f.lower().endswith(e) for e in ext)])
            if img_files:
                if subfolder not in img_dict:
                    img_dict[subfolder] = []
                img_dict[subfolder].extend(img_files)
                num_found_files += len(img_files)
                print(f'Found files: {num_found_files}', end='\r')
        print()
        return img_dict


    def load_images_protocol_two_datasets(self, data_dir='', data_dir_positive_samples=''):
        files_paths = self.get_all_files_in_path(data_dir)
        num_pos_pairs, num_neg_pairs = 0, 0

        # Negative doppelganger pairs
        protocol = int(len(files_paths)/2) * [None]
        for i in range(0, len(files_paths), 2):
            sample0, sample1 = files_paths[i], files_paths[i+1]
            sample0_gender = sample0.split('/')[-2]
            sample1_gender = sample1.split('/')[-2]
            pair_label = 0

            pair = {}
            pair['sample0'] = sample0
            pair['sample0_gender'] = sample0_gender
            
            pair['sample1'] = sample1
            pair['sample1_gender'] = sample1_gender
            
            pair['pair_label'] = pair_label

            protocol[int(i/2)] = pair
            num_neg_pairs += 1

        # Positive pairs
        protocol_pos_pairs = []
        print(f'Loading positive samples: \'{data_dir_positive_samples}\'')
        dict_imgs_paths_by_subj_pos_samples = self.load_dict_imgs_paths_by_subj(data_dir_positive_samples)
        for idx_subj, subj in enumerate(list(dict_imgs_paths_by_subj_pos_samples.keys())):
            samples_subj = dict_imgs_paths_by_subj_pos_samples[subj]
            for idx_sample0 in range(0, len(samples_subj)-1):
                for idx_sample1 in range(idx_sample0+1, len(samples_subj)):
                    sample0 = samples_subj[idx_sample0]
                    sample0_subj = sample0.split('/')[-2]

                    sample1 = samples_subj[idx_sample1]
                    sample1_subj = sample1.split('/')[-2]

                    pair_label = 1

                    pair = {}
                    pair['sample0'] = sample0
                    pair['sample0_subj'] = sample0_subj
                    
                    pair['sample1'] = sample1
                    pair['sample1_subj'] = sample1_subj
                    
                    pair['pair_label'] = pair_label

                    protocol_pos_pairs.append(pair)
                    num_pos_pairs += 1

        # PROVISORY (DUE THE WHOLE DATASET DOESN'T FIT ON MERMORY)
        random.seed(440)
        random.shuffle(protocol_pos_pairs)
        protocol_pos_pairs = protocol_pos_pairs[:len(protocol)]
        num_pos_pairs=len(protocol_pos_pairs)

        protocol.extend(protocol_pos_pairs)
        random.shuffle(protocol)   # shuffle positive and negative pairs

        print(f'num_pos_pairs: {num_pos_pairs}    num_neg_pairs: {num_neg_pairs}')
        assert len(protocol) == num_pos_pairs+num_neg_pairs
        return protocol


    def load_dataset(self, protocol_file, data_dir, data_dir_positive_samples, image_size, replace_ext='.png'):
        print(f"Loading protocol: \'{protocol_file}\'")
        pairs_orig = self.load_protocol(protocol_file)
        pairs_update = self.update_paths(pairs_orig, data_dir, data_dir_positive_samples, replace_ext, inplace=False)

        # pairs_orig = self.load_images_protocol(data_dir)
        pairs_orig = self.load_images_protocol_two_datasets(data_dir, data_dir_positive_samples)
        pairs_update = pairs_orig

        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(pairs_update)*2, 3, image_size[0], image_size[1]))
            data_list.append(data)

        issame_list               = np.array([bool(pairs_update[i]['pair_label']) for i in range(len(pairs_update))])
        # gender_list             = np.array([sorted((pairs_update[i]['sample0_gender'], pairs_update[i]['sample1_gender'])) for i in range(len(pairs_update))])
        samples_orig_paths_list   = np.array([(pairs_orig[i]['sample0'], pairs_orig[i]['sample1']) for i in range(len(pairs_orig))])
        samples_update_paths_list = np.array([(pairs_update[i]['sample0'], pairs_update[i]['sample1']) for i in range(len(pairs_update))])
        
        for idx in range(len(pairs_update) * 2):
            idx_pair = int(idx/2)
            if idx % 2 == 0:
                img_path = pairs_update[idx_pair]['sample0']
            else:
                img_path = pairs_update[idx_pair]['sample1']
            assert os.path.isfile(img_path), f"Error, file not found: '{img_path}'"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mx.nd.array(img)

            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 100 == 0:
                print(f"loading pairs {idx}/{len(pairs_update)*2}", end='\r')
        print('\n', data_list[0].shape)
        # return data_list, issame_list, races_list, subj_list, samples_orig_paths_list, samples_update_paths_list
        # return data_list, issame_list, gender_list, samples_orig_paths_list, samples_update_paths_list
        # return data_list, issame_list, samples_orig_paths_list, samples_update_paths_list
        return {'data_list':data_list,
                'issame_list': issame_list,
                'samples_orig_paths_list': samples_orig_paths_list,
                'samples_update_paths_list': samples_update_paths_list}


