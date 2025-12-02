import copy
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms

from models import *

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'basic_training'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def build_dataloaders(rank, world_size):
    """
    Build distributed CIFAR-10 dataloaders; rank 0 handles download.
    """
    if rank == 0:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        dist.barrier()
    else:
        dist.barrier()
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, sampler=test_sampler, num_workers=4, pin_memory=True)
    return train_loader, test_loader, train_sampler

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    base_model = model.module if isinstance(model, DDP) else model
    model_copied = copy.deepcopy(base_model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


def train(net, train_loader, optimizer, criterion, adversary, epoch, device):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = net(inputs)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign train loss:', loss.item())

    print('\nTotal benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

def test(net, test_loader, criterion, adversary, epoch, device):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {'net': net.module.state_dict() if isinstance(net, DDP) else net.state_dict()}
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if dist.get_rank() == 0:
        torch.save(state, './checkpoint/' + file_name)
        print('Model Saved!')

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    train_loader, test_loader, train_sampler = build_dataloaders(rank, world_size)

    net = ResNet18().to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # Use underlying module for adversarial crafting to avoid DDP wrappers inside attack
    adversary = LinfPGDAttack(net.module if isinstance(net, DDP) else net)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

    for epoch in range(0, 200):
        adjust_learning_rate(optimizer, epoch)
        train_sampler.set_epoch(epoch)
        train(net, train_loader, optimizer, criterion, adversary, epoch, device)
        test(net, test_loader, criterion, adversary, epoch, device)


if __name__ == "__main__":
    main()
