from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cifar10 import Cifar10, ToTensor, Normalize
from k_nearest_pixels import get_k_nearest_image
from model import RNN
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import cv2
from copy import deepcopy
from ssim_loss import SSIM
import numpy as np

root = ''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 3
hidden_size = 8
num_layers = 2
output_size = 3
# batch_size = 16
num_epochs = 50
learning_rate = 0.001
k = 76                              # Give an even number
sequence_length = k + 1
w = 0.5
window_size = 1
test_size = 1000
train_size = 4000
from_checkpoint = True


def unnormalize(img):
    """
    Returns unnormalized numpy version of a tensor. (Warning: Does not support deep copy. Normalizes inplace)
    Args:
        img(tenor): Image to be unnormalized
    Returns: Unnormalized numpy array
    """
    # img_copy = deepcopy(img)
    unnormalized_img = transforms.Normalize((0, 0, 0), (1 / 255.0, 1 / 255.0, 1 / 255.0))(img)
    unnormalized_img = unnormalized_img.permute(1, 2, 0)
    unnormalized_img_np = unnormalized_img.int().numpy()
    return unnormalized_img_np


def generate_img(img, pos, pixels):
    """
    Returns reconstructed image by replacing pixels in img at pos
    Args:
        img(tensor): Image to replace
        pos(tensor): Positions at which pixels are to be replaced
        pixels(tensor): Pixels to replace in image

    Returns: Reconstructed image
    """
    img_copy = deepcopy(img)
    try:
        img_copy[(slice(None), *zip(*pos.long()))] = pixels.cpu()
    except:
        img_copy[(slice(None), *zip(*pos.long()))] = pixels.cpu().t()
    return img_copy


if __name__ == '__main__':
    compose = transforms.Compose(
        [ToTensor(),
         Normalize()
         ])
    print("Extracting Dataset")
    transformed_dataset = Cifar10(cropped=root + 'dataset/cifar-10/cropped_400.npy',
                                  ground_truth=root + 'dataset/cifar-10/original_400.npy',
                                  size=train_size+test_size,
                                  transform=compose
                                  )
    print("Done")
    train_set, test_set = torch.utils.data.random_split(transformed_dataset, [train_size, test_size])
    train_loader, test_loader = DataLoader(train_set, shuffle=False), \
                                DataLoader(test_set, shuffle=False)
    if from_checkpoint:
        model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
        model.load_state_dict(torch.load('models/model30_0.005.pt', map_location=device))
    else:
        model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

    # Loss and optimizer
    loss = nn.L1Loss()
    ssim_loss = SSIM(window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        total_cost = 0
        total_ssim_cost = 0
        total_l1 = 0
        start = time.time()

        for i, sample in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()

            noisy, ground_truth = sample['noisy'], sample['ground_truth']
            k_nearest_pixels, black_pixels = get_k_nearest_image(noisy, k)
            k_nearest_pixels = k_nearest_pixels.to(device)

            outputs = model(k_nearest_pixels[:, :, :3])

            if i == 0:
                reconstructed_img = generate_img(noisy[0], black_pixels, outputs)
                reconstructed_img = unnormalize(reconstructed_img)
                ground = unnormalize(ground_truth[0])
                cv2.imwrite("ckp"+str(epoch)+".jpg", reconstructed_img)
                # print(ground_truth.size())
                cv2.imwrite("ckp.jpg", ground)

                # print(reconstructed_img.shape)
            #
            # if i == 500 or i == 1000 or i == 1500:
                # print('step: {}    cost =  {}   L1 = {}     ssim = {}    time = {}'.format(i, total_cost,total_l1, total_ssim_cost/(512*3*500),
                                                                                  # time.time() - start))
                # reconstructed_img = generate_img(noisy[0], black_pixels, outputs)
                # reconstructed_img = unnormalize(reconstructed_img)
                # cv2.imwrite("ckp.jpg", reconstructed_img)

            reconstructed_img = generate_img(noisy[0], black_pixels, outputs)
            gt = ground_truth[(0, slice(None), *zip(*black_pixels))].t().to(device)
            # print(reconstructed_img.view(noisy.size()).size(), noisy.size())
            # print(ground_truth.size())
            # print(gt.view(noisy.size()))
            L1_cost = w * loss(gt, outputs)
            #             # print(ssim_loss(ground_truth, reconstructed_img.view(noisy.size())).to(device))
            #             ssim_cost = (1-w) * (1 - ssim_loss(ground_truth, reconstructed_img.view(noisy.size())).to(device))
            #             # ssim_cost1 = ssim_loss(ground_truth, ground_truth).to(device)
            #             # print(ssim_cost1)
            #
            #             # print(ssim_cost)
            #
            #             # print(ssim_cost.type())
            cost = L1_cost + ssim_cost
            if i == 0:
                print("cost", cost)
            # print(cost.type())
            cost.backward()
            total_cost += cost
            total_l1 += L1_cost
            total_ssim_cost += ssim_cost
            optimizer.step()
            torch.cuda.empty_cache()

        print('Epoch: {}    total_cost =  {}    L1 = {}     ssim = {}    time = {}'.format(epoch, total_cost, total_l1, total_ssim_cost, time.time() - start))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'model.pt')
    # torch.save(model.state_dict(), '')

    #
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         # print((images.shape, labels.shape))
    #
    #         images = images.reshape(-1, sequence_length, input_size).to(device)
    #
    #         # print(len(images))
    #         labels = labels.to(device)
    #
    #         # Forward pass
    #         outputs = model(images)
    #
    #         loss = criterion(outputs, labels)
    #
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (i + 1) % 100 == 0:
    #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    #                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    # with torch.no_grad():
    #     for i, sample in enumerate(tqdm(test_loader)):
    #
    #         noisy, ground_truth = sample['noisy'], sample['ground_truth']
    #         k_nearest_pixels, black_pixels = get_k_nearest_image(noisy, k)
    #         k_nearest_pixels = k_nearest_pixels.to(device)
    #
    #         outputs = model(k_nearest_pixels[:, :, :3])
    #
    #         reconstructed_img = generate_img(noisy[0], black_pixels, outputs)
    #         reconstructed_img = unnormalize(reconstructed_img)
    #         ground = unnormalize(ground_truth[0])
    #         cv2.imwrite("Outputs/ground_truth_" + str(i) + ".jpg", ground)
    #         cv2.imwrite("Outputs/reconstructed_" + str(i) + ".jpg", reconstructed_img)
    #
    #         if i == 3:
    #              break
        # correct = 0
        # total = 0
        # for images, labels in test_loader:
        #     images = images.reshape(-1, sequence_length, input_size).to(device)
        #     labels = labels.to(device)
        #     outputs = model(images)
        #     print(outputs.data)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
        #
        # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    # torch.save(model.state_dict(), 'model.ckpt')