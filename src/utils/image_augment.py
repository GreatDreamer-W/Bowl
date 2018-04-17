import os

import cv2

from src.utils.transform import do_contrast
from src.utils.utils import get_images


def image_augment(base_path, i):
    images = get_images(base_path)
    for image_id, image in images.items():
        image = cv2.blur(image, (5, 5))
        image = do_contrast(image, 2)
        save_image_augment(image_id, image, i)


def save_image_augment(image_id, image, i):
    # 创建文件夹
    directory = os.path.join("..", "image_augment_" + str(i))
    if not os.path.exists(directory):
        os.mkdir(directory)

    # 存储增强后的图片
    images_dir = os.path.join(directory, image_id + "_augment_" + str(i))
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
        os.mkdir(os.path.join(images_dir, "images"))
    cv2.imwrite(os.path.join(images_dir, "images",
                             image_id + "_augment_" +
                             str(i) + ".png"), image)


def test():
    # images = get_images(base_path)
    # for image_id, image in images.items():
    #     image =
    image = cv2.imread("2.png")
    contrast = do_contrast(image, 3)
    # blur = cv2.blur(contrast, (3, 3))
    cv2.imshow("blur", contrast)
    cv2.waitKey(0)



