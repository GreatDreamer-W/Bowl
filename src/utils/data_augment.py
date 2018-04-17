import os
import random
import numpy as np
import cv2

from src.utils.utils import get_images, get_masks
from src.utils.transform import random_transform, do_color_shift, \
    do_hue_shift, do_saturation_shift, do_decolor, do_custom_process1, do_gamma, \
    do_unsharp, do_inv_speckle_noise, random_transform2, \
    do_elastic_transform2, random_crop_transform2, do_flip_transpose2, \
    do_shift_scale_rotate2

WIDTH, HEIGHT = 200, 200


def train_augment(base_path, num):
    images = get_images(base_path)
    masks = get_masks(base_path)
    for i in range(num):
        for image_id, image in images.items():
            print(image_id)
            image_masks = masks[image_id]
            aug_image, aug_masks = augment(image, image_masks)
            save_augment(aug_image, aug_masks, image_id, i)


def save_augment(aug_image, aug_masks, image_id, i):
    # 创建文件夹
    directory = os.path.join("..", "augment_" + str(i))
    if not os.path.exists(directory):
        os.mkdir(directory)

    # 存储增强后的图片
    images_dir = os.path.join(directory, image_id + "_augment_" + str(i))
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
        os.mkdir(os.path.join(images_dir, "images"))
    cv2.imwrite(os.path.join(images_dir, "images",
                             image_id + "_augment_" +
                             str(i) + ".png"), aug_image)

    # 存储增强后的掩码
    if not os.path.exists(os.path.join(images_dir, "masks")):
        os.mkdir(os.path.join(images_dir, "masks"))
    try:
        for j in range(aug_masks.shape[2]):
            if np.sum(aug_masks[:, :, j]) > 0:
                cv2.imwrite(os.path.join(images_dir, "masks", "augment_" + str(i) +
                                         "_" + str(j) + ".png"), aug_masks[:, :, j])
    except IndexError:
        if np.sum(aug_masks[:, :]) > 0:
            cv2.imwrite(os.path.join(images_dir, "masks", "augment_" + str(i) +
                                     "_" + str(0) + ".png"), aug_masks[:, :])


def augment(image, masks):
    # color
    if 1:
        type = random.randint(0, 4)
        if type == 0:
            image = random_transform(image, u=0.5, func=do_color_shift,
                                     alpha0=[-0.2, 0.2], alpha1=[-0.2, 0.2],
                                     alpha2=[-0.2, 0.2])

        elif type == 1:
            image = random_transform(image, u=0.5, func=do_hue_shift,
                                     alpha=[-0.3, 0.3])

        elif type == 2:
            image = random_transform(image, u=0.5, func=do_saturation_shift,
                                     alpha=[0, 0.3])

        elif type == 3:
            image = random_transform(image, u=0.5, func=do_decolor)

        else:
            pass

    # illumination
    if 1:
        type = random.randint(0, 2)
        if type == 0:
            image = random_transform(image, u=0.5, func=do_custom_process1,
                                     gamma=[0.8, 2.0], alpha=[0.7, 0.9],
                                     beta=[1.0, 1.0])

        elif type == 1:
            image = random_transform(image, u=0.5, func=do_gamma, gamma=[1, 2])

        else:
            pass

    # filter/noise
    if 1:
        type = random.randint(0, 2)
        if type == 0:
            image = random_transform(image, u=0.5, func=do_unsharp,
                                     size=[9, 19], strength=[0.2, 0.4],
                                     alpha=[4, 6])

        elif type == 1:
            image = random_transform(image, u=0.5, func=do_inv_speckle_noise,
                                     sigma=[0.1, 0.2])

        else:
            pass

    # geometric
    if 1:
        image, masks = random_transform2(image, masks, u=0.5,
                                         func=do_shift_scale_rotate2,
                                         dx=[0, 0], dy=[0, 0], scale=[1/2, 2],
                                         angle=[-45, 45])
        image, masks = random_transform2(image, masks, u=0.5,
                                         func=do_elastic_transform2,
                                         grid=[8, 64],
                                         distort=[0, 0.5])

        image, masks = random_crop_transform2(image, masks, WIDTH, HEIGHT, u=0.5)
        image, masks = do_flip_transpose2(image, masks, random.randint(0, 8))

    return image, masks
