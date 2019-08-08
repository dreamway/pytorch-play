import sys
import math
import torch

def get_mean_and_std(dataset):
	''' Compute the mean and std value of dataset. '''
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		for i in range(3): # for 3 channels
			mean[i] += inputs[:,i,:,:].mean()
			std[i] += inputs[:,i,:,:].std()
	mean.div_(len(dataset))
	std.div_(len(dataset))
	return mean, std

"""  # some test code for visualization
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join("%5s"%classes[labels[j]] for j in range(bs)))
"""
