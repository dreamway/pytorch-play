# These code is based on pytorch tutorial (blitz-beginner)
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#functions to show an image
def imshow(img):
    img = img/2+0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

bs = 16
learning_rate = 1e-3
epoches = 100

# transform PILImage of range[0,1] and normalized range [-1, 1]
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.CIFAR10(root="~/data/", train=True,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="~/data/", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')

"""  # some test code for visualization
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join("%5s"%classes[labels[j]] for j in range(bs)))
"""

class Net(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1,16*5*5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train(epoch):
    train_loss = 0.0
    iter_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter_loss += loss.item()
        train_loss += loss.item()
        if (i+1)%2000 == int(len(trainloader)/bs*0.2):
            print('training [epoch {}, iter {}/{}], iter_loss: {:.3f}'.format(epoch+1, i+1, len(trainloader)/bs, iter_loss/2000))
            iter_loss = 0.0
    print('training epoch {}, train_loss: {:.3f}'.format(epoch+1, train_loss/len(trainloader)))
    writer.add_scalar('train_loss', train_loss/len(trainloader), epoch)

def test(epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            c = (predicted==labels).squeeze()
            for j in range(bs):
                label = labels[j]
                class_correct[label.item()] += c[j].item()
                class_total[label.item()] += 1
        
    print('test epoch {}, test loss: {:.3f}'.format(epoch+1, test_loss/len(testloader)))
    print('Accuracy of the network on the test images: {:.1f} %'.format(100.*correct/total))
    for i in range(10):
        print('Accuracy of {}: {:.1f} %'.format(classes[i], 100.*class_correct[i]/class_total[i]))
    writer.add_scalar('test_loss', test_loss/len(testloader))
    writer.add_scalar('test_accuracy', 100.*correct/total)

def main():
    for epoch in range(epoches):
        train(epoch)
        test(epoch)


if __name__ == "__main__":
    main()