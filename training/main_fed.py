#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import torch
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNCifar, VGG
from models.Fed import FedAvg
from models.test import test_img
import numpy as np


def main_fed(dataset_train, dataset_test, dict_users, idxs_users):
    # parse args
    args = args_parser()

    # If use CPU need to set --gpu -1
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'vgg' and args.dataset == 'cifar':
        net_glob = VGG().to('cpu')
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    accuracy_train = []
    accuracy_test = []
    
    for iter in range(args.epochs):
        # every 30 epoches, fixed users
        if iter % 30 == 0:
            idxs_users = np.random.choice(range(100), 10, replace=False)
        print(f'chosen workers {idxs_users}')
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # testing the current model on dataset_train and dataset_test
        net_glob.eval()
        # Test sampled training data accuracy
        # Get all sampled training data index
        train_data_index = []
        for idx in idxs_users:
            for image_idx in dict_users[idx]:
                train_data_index.append(image_idx)
        acc_train, _, dict_correct_train = test_img(net_glob, dataset_train, args, train_data_index)
        acc_test, loss_test, dict_correct_test = test_img(net_glob, dataset_test, args, np.arange(len(dataset_test)))
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        accuracy_train.append(acc_train)
        accuracy_test.append(acc_test)

    return accuracy_train, accuracy_test