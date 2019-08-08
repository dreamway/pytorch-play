# These code is based on pytorch tutorial (blitz-beginner)
import os
import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import *
from torch.optim.lr_scheduler import *
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="cifar10 pytorch play")
parser.add_argument('--lr', dest='lr', type=float, default=0.1, help="set the learning rate")
parser.add_argument('--epochs', dest='epochs', type=int, default=300, help="the epochs")
parser.add_argument('--resume', dest="resume", action="store_true", default=False, help="resume training")
parser.add_argument('--bs', dest="bs", type=int, default=128, help="set the batch size")
args = parser.parse_args()

writer = SummaryWriter()

#functions to show an image
def imshow(img):
    img = img/2+0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

bs = args.bs
learning_rate = args.lr
epoches = args.epochs
best_acc = 0 
start_epoch = 0
print('the input parameters: bs=', bs,", lr=", learning_rate, ', epochs=',epoches)

mean = (0.4914,0.4822,0.4465)
std = (0.2023, 0.1994, 0.2010)
# transform PILImage of range[0,1] and normalized range [-1, 1]
transform_train = transforms.Compose(
	[ transforms.RandomCrop(32, padding=4),
	  transforms.RandomHorizontalFlip(),
	  transforms.ToTensor(),
	  transforms.Normalize(mean, std)
   	])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)
    ])

trainset = torchvision.datasets.CIFAR10(root="~/data/", train=True,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root="~/data/", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')

model = resnet18()
model.to(device)

if args.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint...')
    assert osp.isdir('checkpoint'), 'Error: not checkpoint dir!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,350], gamma=0.1)


def train(epoch):
    train_loss = 0.0
    iter_loss = 0.0
    model.train()
    correct = 0
    total = 0
    print('start training, epoch: ', epoch)
    for batch_idx, (images, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (batch_idx+1)%200 == 0:
            print('training [epoch {}, iter {}/{}], train_loss: {:.3f}'.format(epoch+1, batch_idx, len(trainloader), train_loss/(batch_idx+1)))
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print('training epoch {}, train_loss: {:.3f}'.format(epoch+1, train_loss/len(trainloader)))
    writer.add_scalar('train_loss', train_loss/len(trainloader), epoch)
    accuracy = correct*100./total
    writer.add_scalar('train_acc', accuracy, epoch)
    print('training epoch{}, accuracy: {}'.format(epoch+1, accuracy))


def test(epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    global best_acc
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    print('evaluating, epoch:', epoch)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            """
            c = (predicted==labels).squeeze()
            for j in range(bs):
                label = labels[j]
                class_correct[label.item()] += c[j].item()
                class_total[label.item()] += 1
            """
            if (batch_idx+1)%100==0:
                print("test epoch {}, iter{}/{}, loss:{:.2f}".format(epoch, batch_idx, len(testloader), test_loss/(batch_idx+1)))
        
    print('test epoch {}, test loss: {:.3f}'.format(epoch+1, test_loss/len(testloader)))
    accuracy = 100.*correct/total
    print('Accuracy of the network on the test images: {:.1f} %'.format(accuracy))
    """
    for i in range(10):
        print('Accuracy of {}: {:.1f} %'.format(classes[i], 100.*class_correct[i]/class_total[i]))
    """
    writer.add_scalar('test_loss', test_loss/len(testloader), epoch)
    writer.add_scalar('test_accuracy', accuracy, epoch)
   
    # save the checkpoints
    if accuracy > best_acc:
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        print('saving...')
        state = {
            'net': model.state_dict(),
            'acc': accuracy,
            'epoch': epoch
        }
        torch.save(state, './checkpoint/ckpt.pth')        
        best_acc = accuracy        

    return test_loss


def main():
    for epoch in range(start_epoch, start_epoch+epoches):
        train(epoch)
        val_loss = test(epoch)
        scheduler.step()
    
    writer.close()

if __name__ == "__main__":
    main()
