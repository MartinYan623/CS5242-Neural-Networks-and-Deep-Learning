import pickle
import random
import numpy as np
from kdtree import *
from read_pdb_file import read_pdb
from tqdm import tqdm

NUM_NEG = 2
NUM_NEAR = 4


def extract_data(i, type):
    x_list, y_list, z_list, atomtype_list = read_pdb('../data/training_data/%s_%s_cg.pdb' % (str(i).zfill(4), type))
    data = np.transpose(np.array([x_list, y_list, z_list, atomtype_list]))
    return np.array(data)


def find_mean_point(data):
    '''

    :param data: [[],[],[]]
    :return: []
    '''
    mean_v = np.mean(data, axis=0)
    vec_o = np.delete(mean_v, 3)
    return vec_o


def transform_data(data, meanpoint):
    transform_data = data[:, :3] - meanpoint

    # scaling_data = np.trunc(transform_data / 3).astype(np.int)
    return transform_data


def transform_data_tree(data, meanpoint):
    mean = np.append(meanpoint, 0)
    t = data - mean
    return t


def prepare_CNN(index, type):
    if type is 'pro':
        pro = extract_data(index, 'pro')
        origin_point_pro = find_mean_point(pro)
        pro = transform_data(pro, origin_point_pro)
        # print(origin_point_pro)
        cnn_pro = np.zeros((27, 27, 27, 1))
        for atom in pro:
            location = list(map(lambda x: int(x / 5), atom))
            x = location[0]
            y = location[1]
            z = location[2]
            if x in range(-13, 14) and y in range(-13, 14) and z in range(-13, 14):
                cnn_pro[x][y][z][0] += 1
            else:
                pass
        return cnn_pro
    elif type is 'lig':
        lig = extract_data(index, 'lig')
        origin_point_lig = find_mean_point(lig)
        lig = transform_data(lig, origin_point_lig)
        # print(origin_point_lig)
        cnn_lig = np.zeros((6, 6, 6, 1))
        for atom in lig:
            location = list(map(lambda x: int(x / 5), atom))
            x = location[0]
            y = location[1]
            z = location[2]
            if x in range(-3, 3) and y in range(-3, 3) and z in range(-3, 3):
                cnn_lig[x][y][z][0] += 1
            else:
                pass
        return cnn_lig
    else:
        print('Wrong type!')


def sample_neg(num, i, N):
    # Sample n different ligs
    flag = True
    while flag:
        a = range(1, N+1)
        neg_sample = random.sample(a, num)
        if i not in neg_sample:
            flag = False
    return neg_sample


def create_CNN_train(num):
    cnn_pro_train = []
    cnn_lig_train = []
    cnn_out_train = []

    for i in tqdm(range(1, num+1)):
        cnn_pro_train.append(prepare_CNN(i, 'pro'))
        cnn_lig_train.append(prepare_CNN(i, 'lig'))
        cnn_out_train.append([1])

        indexes_neg = sample_neg(NUM_NEG, i, 2700)
        for sample in indexes_neg:
            cnn_pro_train.append(prepare_CNN(i, 'pro'))
            cnn_lig_train.append(prepare_CNN(sample, 'lig'))
            cnn_out_train.append([-1])

    with open ('../data/cnn_data/cnn_pro_train.bin', 'wb') as f:
        pickle.dump(cnn_pro_train, f)
    with open ('../data/cnn_data/cnn_lig_train.bin', 'wb') as f:
        pickle.dump(cnn_lig_train, f)
    with open ('../data/cnn_data/cnn_out_train.bin', 'wb') as f:
        pickle.dump(cnn_out_train, f)
    print('\nCNN training data stored successfully!\n')


def create_CNN_valid(num1, num2):
    cnn_pro_valid = []
    cnn_lig_valid = []
    cnn_out_valid = []

    print('Begin storing valid dataset')
    for i in tqdm(range(num1+1, num2+1)):
        for j in range(num1+1, num2+1):
            cnn_pro_valid.append(prepare_CNN(i, 'pro'))
            cnn_lig_valid.append(prepare_CNN(j, 'lig'))
            if i == j:
                cnn_out_valid.append([1])
            else:
                cnn_out_valid.append([-1])

    with open ('../data/cnn_data/cnn_pro_valid.bin', 'wb') as f:
        pickle.dump(cnn_pro_valid, f)
    with open ('../data/cnn_data/cnn_lig_valid.bin', 'wb') as f:
        pickle.dump(cnn_lig_valid, f)
    with open ('../data/cnn_data/cnn_out_valid.bin', 'wb') as f:
        pickle.dump(cnn_out_valid, f)
    print('\nCNN training data stored successfully!\n')



def store_tree():
    tree_list = []
    with open('../data/middle_data/tree_list.bin', 'wb') as f:
        for i in tqdm(range(1, 3000 + 1)):
            pro = extract_data(i, 'pro')
            origin_point_pro = find_mean_point(pro)
            pro = transform_data_tree(pro, origin_point_pro)

            tree_list.append(build_KDTree(pro))
        pickle.dump(tree_list, f)
        
    print('Info stored successfully!')


def create_mlp_train(tree_list, N):
    # training data: 1 matched pair and 5 random unmatched pairs
    train_input = []
    train_output = []
    for i in tqdm(range(1, N + 1)):
        pro = extract_data(i, 'pro')
        # get pro centroid
        origin_point_pro = find_mean_point(pro)
        # recompute lig coordinates
        lig = extract_data(i, 'lig')
        lig = transform_data_tree(lig, origin_point_pro)

        train_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], lig, NUM_NEAR))
        train_output.append([1])

        neg_samples = sample_neg(NUM_NEG, i, N)
        for j in neg_samples:
            neg_lig = extract_data(j, 'lig')
            neg_lig = transform_data_tree(neg_lig, origin_point_pro)
            train_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], neg_lig, NUM_NEAR))
            train_output.append([-1])

    with open('../data/middle_data/train_input.bin', 'wb') as f:
        pickle.dump(train_input, f)
    with open('../data/middle_data/train_output.bin', 'wb') as f:
        pickle.dump(train_output, f)
    print('\ntraining data stored successfully!\n')


def create_mlp_valid(tree_list, N):
    # validation data: store every pair
    valid_input = []
    valid_output = []
    for i in tqdm(range(N + 1, 3001)):
        pro = extract_data(i, 'pro')
        # get pro centroid
        origin_point_pro = find_mean_point(pro)

        for j in range(N + 1, 3001):
            # recompute lig coordinates
            lig = extract_data(j, 'lig')
            lig = transform_data_tree(lig, origin_point_pro)
            valid_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], lig, NUM_NEAR))

            if i == j:
                valid_output.append([1])
            else:
                valid_output.append([-1])

    print('\nvalidation data constructed successfully!\n')
    return valid_input, valid_output


if __name__ == '__main__':


    store_tree()
    """
    with open('../data/middle_data/tree_list.bin', 'rb') as f:
         tree_list = pickle.load(f)
    print('Tree info loaded successfully!')

    create_mlp_train(tree_list, 2700)
    create_mlp_valid(tree_list, 2700)

    create_CNN_train(3000)
    create_CNN_valid(2990, 3000)

    with open('../data/middle_data/train_input.bin', 'rb') as f:
        train_input = pickle.load(f)
    with open('../data/middle_data/train_output.bin', 'rb') as f:
        train_output = pickle.load(f)
    with open('../data/middle_data/valid_input.bin', 'rb') as f:
        valid_input = pickle.load(f)
    with open('../data/middle_data/valid_output.bin', 'rb') as f:
        valid_output = pickle.load(f)
    print('Data info loaded successfully!')

    for i in range(len(valid_input)):
        print(i)
        print(valid_input[i])
        if i > 5:
            break

    pass

    print('\n\n*****************************************\nThe KDTree answer is:\n')
    output = find_nearest_atoms_KDTree(t, lig, 3)
    print(np.array(output))

    print("\n\n*****************************************\nThe YMT's answer is:\n")
    print(find_nearest_atoms_YMT(pro, lig, 3))
    """








