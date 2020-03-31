import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
root = ''
split = 0.8


class Cifar10(Dataset):

    def __init__(self, cropped, ground_truth, size=5000, transform=None):
        """
        Args:
            cropped (string): File location of cropped numpy
            ground_truth (string): File location of ground_truth numpy
            size (int, optional): Size of dataset to be used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.noisy = np.load(cropped)[0: size]
        self.ground_truth = np.load(ground_truth)[0: size]

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        noisy = self.noisy[idx].transpose(2, 0, 1)
        ground_truth = self.ground_truth[idx].transpose(2, 0, 1)
        sample = {'noisy': noisy, 'ground_truth': ground_truth}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        noisy, ground_truth = sample['noisy'], sample['ground_truth']
        return {'noisy': torch.from_numpy(noisy).float(),
                'ground_truth': torch.from_numpy(ground_truth).float()}


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, transform_=transforms.Normalize((0, 0, 0), (255.0, 255.0, 255.0))):
        noisy, ground_truth = transform_(sample['noisy']), transform_(sample['ground_truth'])
        return {'noisy': noisy,
                'ground_truth': ground_truth}


# if __name__ == '__main__':
#     compose = transforms.Compose(
#         [ToTensor(),
#          Normalize()
#          ])
#     transformed_dataset = Cifar10(cropped=root + 'cropped.npy',
#                                   ground_truth=root + 'original.npy',
#                                   transform=compose)
#     # with open('transformed_dataset.pkl', 'wb') as f:
#     #     pickle.dump(transformed_dataset, f)
#
#     dataloader = DataLoader(transformed_dataset, batch_size=1,
#                             shuffle=True)
#     # print(transformed_dataset)
#     # print(len(dataloader))
#
#     for x in dataloader:
#         print(x['noisy'])
#         print(x['ground_truth'])
#         print(torch.unique(x['noisy']))
