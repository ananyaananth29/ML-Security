import os

import torch
import torch as ch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms

from models import *

learning_rate = 0.1
file_name = 'basic_training_with_robust_dataset'

transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)

def build_dataloaders(rank, world_size):
    """
    Build distributed loaders; rank 0 handles CIFAR-10 download.
    """
    data_path = "madry_data/release_datasets/d_robust_CIFAR/"
    train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
    train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
    train_dataset = TensorDataset(train_data, train_labels, transform=transform_train)

    if rank == 0:
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        dist.barrier()
    else:
        dist.barrier()
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, sampler=test_sampler, num_workers=4, pin_memory=True)
    return train_loader, test_loader, train_sampler


def l2_pgd_attack(model, images, labels, device, eps=0.25, alpha=0.01, steps=20):
    """
    Simple L2 PGD attacker compatible with current torch; avoids advertorch dependency.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    ori = images.clone().detach()

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = images.grad.detach()

        grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
        grad_norm = grad_norm.view(-1, 1, 1, 1)
        normalized_grad = grad / (grad_norm + 1e-12)

        images = images + alpha * normalized_grad

        delta = images - ori
        delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1, keepdim=True)
        delta_norm = delta_norm.view(-1, 1, 1, 1)
        factor = torch.clamp(delta_norm, max=eps)
        delta = delta * (eps / (factor + 1e-12))
        images = torch.clamp(ori + delta, 0, 1).detach()

    return images

def train(net, train_loader, optimizer, criterion, epoch, device):
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

def test(net, test_loader, criterion, epoch, device):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
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

        adv = l2_pgd_attack(net, inputs, targets, device, eps=0.25, alpha=0.01, steps=20)
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
    if dist.get_rank() == 0:
        os.makedirs('checkpoint', exist_ok=True)
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

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

    for epoch in range(0, 200):
        adjust_learning_rate(optimizer, epoch)
        train_sampler.set_epoch(epoch)
        train(net, train_loader, optimizer, criterion, epoch, device)
        test(net, test_loader, criterion, epoch, device)


if __name__ == "__main__":
    main()
