from copy import deepcopy
from torchvision import transforms
import cv2


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


def checkpoint(noisy_img, ground_truth_img, black_pixels, outputs, epoch):
    reconstructed_img = generate_img(noisy_img, black_pixels, outputs)
    reconstructed_img = unnormalize(reconstructed_img)
    ground = unnormalize(ground_truth_img)
    cv2.imwrite("ckp" + str(epoch) + ".jpg", reconstructed_img)0- e
    cv2.imwrite("ckp.jpg", ground)
