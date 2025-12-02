import os

import torch
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


# --------------------
# SETTINGS
# --------------------
learning_rate = 0.1
file_name = 'pgd_training_with_robust_dataset'

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# --------------------
# Dataloader for robust dataset
# --------------------
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
    Construct distributed dataloaders after process group is initialized.
    Rank 0 handles CIFAR-10 download to avoid redundant work.
    """
    data_path = "madry_data/release_datasets/d_robust_CIFAR/"
    train_data = torch.cat(torch.load(os.path.join(data_path, "CIFAR_ims")))
    train_labels = torch.cat(torch.load(os.path.join(data_path, "CIFAR_lab")))

    train_dataset = TensorDataset(train_data, train_labels, transform=transform_train)
    if rank == 0:
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform_test)
        # Allow other ranks to proceed once data is present
        dist.barrier()
    else:
        # Wait for rank 0 to finish downloading, then load without download flag
        dist.barrier()
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=False, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, sampler=test_sampler,
                             num_workers=4, pin_memory=True)
    return train_loader, test_loader, train_sampler

# -------------------------------------------------
# Custom PGD L∞ attack (works with all PyTorch versions)
# -------------------------------------------------
def pgd_attack(model, images, labels, device, eps=8/255, alpha=2/255, steps=10):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    ori_images = images.clone().detach()

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = images.grad.sign()

        # update images
        images = images + alpha * grad
        # projection step
        eta = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images

def train(net, train_loader, optimizer, criterion, epoch, device):
    print(f"\n[ Train epoch: {epoch} ]")
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # ---- PGD adversarial examples ----
        adv_inputs = pgd_attack(net, inputs, targets, device, eps=8/255, alpha=2/255, steps=10)
        outputs = net(adv_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Train Acc: {100.*correct/total:.2f}%")

    print(f"Epoch Train Acc: {100.*correct/total:.2f}%")
    print(f"Epoch Train Loss: {train_loss:.4f}")

def test(net, test_loader, epoch, device):
    print(f"\n[ Test epoch: {epoch} ]")
    net.eval()
    benign_correct = 0
    adv_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        # benign
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        # adversarial
        adv = pgd_attack(net, inputs, targets, device, eps=8/255, alpha=2/255, steps=20)
        adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

    print(f"Clean Test Accuracy: {100.*benign_correct/total:.2f}%")
    print(f"Adversarial Test Accuracy: {100.*adv_correct/total:.2f}%")

    # Save checkpoint
    state = {'net': net.module.state_dict() if isinstance(net, DDP) else net.state_dict()}
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    if dist.get_rank() == 0:
        torch.save(state, './checkpoint/' + file_name + '.pth')
        print("Model Saved!")

# --------------------
# LEARNING RATE SCHEDULER
# --------------------
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# --------------------
# MAIN LOOP
# --------------------
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
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=0.0002)

    for epoch in range(0, 200):
        adjust_learning_rate(optimizer, epoch)
        train_sampler.set_epoch(epoch)
        train(net, train_loader, optimizer, criterion, epoch, device)
        test(net, test_loader, epoch, device)


if __name__ == "__main__":
    main()
