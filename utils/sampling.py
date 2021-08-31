import numpy as np
from collections import defaultdict

def cifar_iid(dataset, num_users=100):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def sort_data(dataset, total_data_num=50000):
    dict_data = defaultdict(list)
    train_labels = dataset.train_labels
    for idx in range(total_data_num):
        label = train_labels[idx]
        dict_data[label].append(idx)
    return dict_data

# Split CIFAR into 100 users, each user has 10 categoreis, and 50 images in each category 
def cifar_split_evenly(dataset, num_users=100, categories=10):
    num_items = int(len(dataset)/num_users/categories)
    dict_users= {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_data = sort_data(dataset)
    for i in range(num_users):
        for j in range(categories):
            rand_set = np.random.choice(dict_data[j], num_items, replace=False)
            dict_data[j] = list(set(dict_data[j]) - set(rand_set))
            dict_users[i] = np.concatenate((dict_users[i], rand_set), axis=0)
    return dict_users

    