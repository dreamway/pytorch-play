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
from models.resnet import *
from torch.optim.lr_scheduler import *
import argparse
from tqdm import tqdm
from models.mobilenet import * #modified version for CIFAR10


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
epochs = args.epochs
best_acc = 0 
start_epoch = 0
best_loss = 1000.0
print('the input parameters: bs=', bs,", lr=", learning_rate, ', epochs=',epochs)

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="~/data/", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

visloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')

model = mobilenet_v2(num_classes=10, pretrained=False)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250,350], gamma=0.1)


#Resume
if args.resume:
    print('==> Resuming from checkpoint...')
    assert osp.isdir('checkpoint'), 'Error: not checkpoint dir!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_acc = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    print("resume from epoch {}, lr {}, when accuracy {}, loss: {}".format(start_epoch, learning_rate, best_acc, best_loss))

print("Model's state_dict:")
for param_tensor in model.state_dict():
	print(param_tensor, '\t', model.state_dict()[param_tensor].size())
print('optimizer:', optimizer)
print("Optimizer's state_dict")
for var_name in optimizer.state_dict():
	print(var_name,"\t", optimizer.state_dict()[var_name])



def train(epoch):
    model.train()
    train_loss = 0.0
    iter_loss = 0.0    
    correct = 0
    total = 0

    with tqdm(total=len(trainloader)) as pbar:
        pbar.set_description("Train epoch:%d"%epoch)
        for batch_idx, (images, labels) in enumerate(trainloader):
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
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            accuracy = correct*100./total            

            postfix_str = "loss:{:.3f}, accuracy:{:.3f}".format(train_loss/total, accuracy)
            pbar.set_postfix_str(postfix_str)
            pbar.update()
   
        writer.add_scalar('train_loss', train_loss/len(trainloader), epoch)
        writer.add_scalar('train_acc', accuracy, epoch)


def test(epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    global best_acc
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        with tqdm(total=len(testloader)) as pbar:
            pbar.set_description("test epoch:%d"%epoch)
            for batch_idx, (inputs, targets) in enumerate(testloader):
                images, labels = inputs.to(device), targets.to(device)

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
                pbar.set_postfix_str("loss:{:.3f}, accuracy:{:.3f}".format(test_loss/total, correct*100./total))
                pbar.update()
        
    accuracy = correct*100./len(testset)
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
	    
        print('saving best for inference...')
        torch.save(model.state_dict(), './checkpoint/model_best.pth')

        print('saving best checkpoint for training resume...')
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'accuracy': accuracy,
        }
        torch.save(state, './checkpoint/ckpt.pth')        
        best_acc = accuracy        

    return test_loss


def main():
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        val_loss = test(epoch)
        scheduler.step()
    
    writer.close()

if __name__ == "__main__":
    main()
