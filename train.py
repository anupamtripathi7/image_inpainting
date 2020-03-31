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
from ssim_loss import SSIM

root = ''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 3
hidden_size = 8
num_layers = 2
output_size = 3
num_epochs = 50
learning_rate = 0.001
k = 76                              # Give an even number
sequence_length = k + 1
w = 0.5
window_size = 1
test_size = 1000
train_size = 4000
from_checkpoint = True


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
                
            if i == 500 or i == 1000 or i == 1500:
                print('step: {}\tcost =  {}\tL1 = {}\tssim = {}\ttime = {}'.format(i, total_cost, total_l1, 
                                                                                   total_ssim_cost/(512*3*500),
                                                                                   time.time() - start))
                reconstructed_img = generate_img(noisy[0], black_pixels, outputs)
                reconstructed_img = unnormalize(reconstructed_img)
                cv2.imwrite("ckp.jpg", reconstructed_img)

            reconstructed_img = generate_img(noisy[0], black_pixels, outputs)
            gt = ground_truth[(0, slice(None), *zip(*black_pixels))].t().to(device)
            L1_cost = w * loss(gt, outputs)
            ssim_cost = (1-w) * (1 - ssim_loss(ground_truth, reconstructed_img.view(noisy.size())).to(device))
            cost = L1_cost + ssim_cost

            cost.backward()
            total_cost += cost.double()
            total_l1 += L1_cost.double()
            total_ssim_cost += ssim_cost.double()
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