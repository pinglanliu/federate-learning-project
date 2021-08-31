from torchvision import datasets, transforms
from training.main_fed import main_fed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.options import args_parser
from utils.sampling import cifar_iid
import sys

# command python test_iid.py --gpu -1 --dataset cifar --num_channels 3 --model cnn --epochs 300

args = args_parser()
sys.stdout = Logger("./logs/cifar_iid.log")

trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

print("Sampled IID workers")
dict_users = cifar_iid(dataset_train)
idxs_users = [i for i in range(10)]
rand_iid_train_accuracy, rand_iid_test_accuray = main_fed(dataset_train, dataset_test, dict_users, idxs_users)
plt.figure()
plt.plot(range(len(rand_iid_train_accuracy)), rand_iid_train_accuracy)
plt.xlabel('rounds')
plt.ylabel('train accuracy')
plt.plot(len(rand_iid_train_accuracy)-1, rand_iid_train_accuracy[-1], 'r*')
plt.annotate(f'{rand_iid_train_accuracy[-1]}', (len(rand_iid_train_accuracy)-1,rand_iid_train_accuracy[-1]))
plt.savefig(f'./results/cifar/iid/fixed-users/train_accuracy_{args.epochs}_{args.model}_{len(idxs_users)}.png')

plt.figure()
plt.plot(range(len(rand_iid_test_accuray)), rand_iid_test_accuray)
plt.xlabel('rounds')
plt.ylabel('test accuracy')
plt.plot(len(rand_iid_test_accuray)-1, rand_iid_test_accuray[-1], 'r*')
plt.annotate(f'{rand_iid_test_accuray[-1]}', (len(rand_iid_test_accuray)-1,rand_iid_train_accuracy[-1]))
plt.savefig(f'./results/cifar/iid/fixed-users/test_accuracy_{args.epochs}_{args.model}_{len(idxs_users)}.png')