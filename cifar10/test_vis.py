from vis.vanilla_backprop import *
import torch
import torchvision
import numpy as np
from models.resnet import *
import torchvision.transforms as transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.201)

transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
testset = torchvision.datasets.CIFAR10(root="~/data/", train=False, download=True, transform=transform_test)
visloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
model = resnet18(pretrained=False, num_classes=10)
model.to(device)

checkpoint = torch.load('./checkpoint/ckpt.pth')
model.load_state_dict(checkpoint['net'])

VBP = VanillaBackprop(model)

def visualize():
    for batch_i, (image, label) in enumerate(visloader):
        print('label:', label)
        image, label = image.to(device), label.to(device)
        vanilla_grads = VBP.generate_gradients(image, label)
        save_gradient_images(vanilla_grads, "test_vbp")

        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        save_gradient_images(grayscale_vanilla_grads, "test_vbp_gray")
        break
        
if __name__ == '__main__':
    visualize()
    