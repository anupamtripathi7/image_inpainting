import time
from tqdm import tqdm
import numpy as np
import torch


def sort_coordinates(coordinates):
    """
    Sorts the given coordinate array in a zig zag manner
    Args:
        coordinates(Tensor): 2d list of coordinates
    Returns: Sorted Tensor
    """
    coordinates = np.array(coordinates)

    odds = coordinates[coordinates[:, 0] % 2 != 0]
    evens = coordinates[coordinates[:, 0] % 2 == 0]
    odds = odds[np.argsort(odds[:, 1])]
    evens = evens[np.argsort(evens[:, 1])[::-1]]
    coordinates = np.concatenate((odds, evens), axis=0)
    coordinates = coordinates[np.argsort(coordinates[:, 0], kind="mergesort")]
    return torch.from_numpy(coordinates).float()


def get_k_nearest_pixels(image, pixel_idx1, pixel_idx2, k=200):
    """
    Returns k nearest non black pixels from (pixel_idx1, pixel_idx2)
    Args:
        image (Tensor): Entire image of size (128, 3, 32, 32)
        pixel_idx1 (Tensor): x co-ordinate of source pixel
        pixel_idx2 (Tensor): y co-ordinate of source pixel
        k (int): Number of nearest pixels needed

    Returns:
        k_nearest_pixels (Tensor): tensor of format (R, G, B, x, y) where x, y is the coordinate of the neighboring pixels
    """
    non_black_pixels = (torch.sum(image, dim=0) != 0).nonzero().float()
    manhattan = torch.sum(torch.abs(non_black_pixels - torch.FloatTensor([pixel_idx1, pixel_idx2])), dim=1)

    non_black_pixels = non_black_pixels[torch.argsort(manhattan)]
    non_black_sorted_k_nearest = non_black_pixels[:min(k, non_black_pixels.size(0))]

    non_black_sorted = sort_coordinates(non_black_sorted_k_nearest)

    channels = image[(slice(None), *zip(*non_black_sorted))].t()

    co_ordinates = torch.cat((non_black_sorted, torch.FloatTensor([[pixel_idx1, pixel_idx2]])), dim=0)      # append black pixel coordinates
    channels = torch.cat((channels, image[:, pixel_idx1.long(), pixel_idx2.long()].view(1, -1)), dim=0)     # append black pixel channel values
    return torch.cat((channels, co_ordinates), dim=1)


def get_k_nearest_image(img, k=200):
    """
    Returns k nearest pixels to all the black pixels in the image
    Args:
        img (Tensor): Batch of images, of size (128, 3, 32, 32)
        k (int): Number of nearest pixels needed

    Returns:
        nearest_pixels (Tensor): K nearest for all black pixel
        black_pixels (Tensor): Co-ordinates of the black pixels
    """

    nearest_pixels = []
    img = img[0]
    black_pixels = (torch.sum(img, dim=0) == 0).nonzero().float()
    for pxl_i, pxl_j in tqdm(black_pixels):
        nearest_pixels.append(get_k_nearest_pixels(img, pxl_i, pxl_j, k))
    return torch.stack(nearest_pixels), black_pixels


if __name__ == '__main__':
    start = time.time()
    x = np.append(np.ones(8192), np.zeros(8192)).reshape((128, 128))
    np.random.shuffle(x)

    x = np.array([np.array(x), np.array(x), np.array(x)])
    # m = np.array([x, x])
    m = torch.from_numpy(x).float()
    print(m.size())

    a, b = get_k_nearest_image(m.view(1, 3, 128, 128), k=200)
    print(a.size())
    print(time.time() - start)
