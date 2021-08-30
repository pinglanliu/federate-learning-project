import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from models.Update import DatasetSplit

def test_img(net_g, datatest, args, idxs):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    dict_correct = defaultdict(float)
    # Test over all images inside the dataset
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    # Test over a selected/sampled images with index inside idxs
    data_loader = DataLoader(DatasetSplit(datatest, idxs), batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        for i in range(len(target)):
            val = target[i]
            if val == y_pred[i]:
                dict_correct[val] += 1
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss, dict_correct

