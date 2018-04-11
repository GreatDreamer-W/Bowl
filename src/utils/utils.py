import numpy as np
import os
import re
from glob import glob
import cv2


def get_image_paths(base_path):
    return glob(os.path.join("..", base_path, "**", "images", "*.png"))


def get_images(base_path):
    print("从" + base_path + "中导入图片...")
    image_paths = get_image_paths(base_path)
    image_id_extractor = re.compile(".*\\\\" + base_path +
                                    "\\\\(?P<image_id>.*)" +
                                    "\\\\images\\\\.*")
    images = dict()
    for image_path in image_paths:
        matches = image_id_extractor.match(image_path)
        image_id = matches.group("image_id")
        image = cv2.imread(image_path)
        images[image_id] = image
    print("成功导入" + str(images.__len__()) + "张图片！")
    return images


def get_masks(base_path):
    print("从" + base_path + "中导入掩码...")
    image_paths = get_image_paths(base_path)
    image_id_extractor = re.compile(".*\\\\" + base_path +
                                    "\\\\(?P<image_id>.*)" +
                                    "\\\\images\\\\.*")
    count = 0
    masks = dict()
    for image_path in image_paths:
        matches = image_id_extractor.match(image_path)
        image_id = matches.group("image_id")
        mask_paths = glob(os.path.join("..", base_path, image_id, "masks",
                                       "*.png"))
        image_masks = []
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, 0)
            image_masks.append(mask)
            count += 1
        masks[image_id] = np.transpose(np.array(image_masks), (1, 2, 0))
    print("成功导入" + str(count) + "张掩码！")
    return masks


def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
