import random
from skimage import measure
import cv2
import numpy as np


# customize augmentation
def do_custom_process1(image, gamma=2.0, alpha=0.8, beta=2.0):
    image1 = image.astype(np.float32)
    image1 = image1 ** gamma
    image1 = image1 / image1.max() * 255

    image2 = alpha * image1 + (1 - alpha) * image
    image2 = np.clip(beta * image2, 0, 255).astype(np.uint8)

    image = image2
    return image


# make random transform
def random_transform(image, u, func, **kwargs):
    if random.random() < u:
        limits = []
        for k in kwargs:
            limit = kwargs[k]
            l = random.uniform(limit[0], limit[1])
            limits.append(l)
        image = func(image, *limits)
    return image


def random_transform2(image, mask, u, func, **kwargs):
    if random.random() < u:
        limits = []
        for k in kwargs:
            limit = kwargs[k]
            l = random.uniform(limit[0], limit[1])
            limits.append(l)
        image, mask = func(image, mask, *limits)
    return image, mask


# illumination
def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha * image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray = image * np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha * image + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image,
                   table)  # apply gamma correction using the lookup table


def do_clahe(image, clip=2, grid=16):
    grid = int(grid)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(
        gray)
    lab = cv2.merge((gray, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


# color
def do_decolor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def do_saturation_shift(image, alpha=1):
    gray = image * np.array([[[0.114, 0.587, 0.299]]])
    gray = np.sum(gray, axis=2, keepdims=True)
    image = alpha * image + (1.0 - alpha) * gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def do_hue_shift(image, alpha=0):
    h = int(alpha * 180)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/53940
def do_color_shift(image, alpha0=0, alpha1=0, alpha2=0):
    image = image.astype(np.float32) + np.array([alpha0, alpha1, alpha2]) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# filter
def do_unsharp(image, size=9, strength=0.25, alpha=5):
    image = image.astype(np.float32)
    size = 1 + 2 * (int(size) // 2)
    strength = strength * 255
    blur = cv2.GaussianBlur(image, (size, size), strength)
    image = alpha * image + (1 - alpha) * blur
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# noise
def do_gaussian_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32) / 255
    H, W = gray.shape

    noise = np.random.normal(0, sigma, (H, W))
    noisy = gray + noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32) / 255
    H, W = gray.shape

    noise = sigma * np.random.randn(H, W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def do_inv_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32) / 255
    H, W = gray.shape

    noise = sigma * np.random.randn(H, W)
    noisy = gray + (1 - gray) * noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


# geometric
def relabel_mask(masks):
    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        data = mask[:, :, np.newaxis]
        unique_color = set(tuple(v) for m in data for v in m)

        H, W = data.shape[:2]
        mask = np.zeros((H, W), np.int32)
        for color in unique_color:
            # print(color)
            if color == (0,): continue

            m = (data == color).all(axis=2)
            label = measure.label(m)

            index = [label != 0]
            mask[index] = label[index] + mask.max()
    return masks


def do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=0):
    border_mode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width, channel = image.shape
    sx = scale
    sy = scale
    import math
    cc = math.cos(angle / 180 * math.pi) * (sx)
    ss = math.sin(angle / 180 * math.pi) * (sy)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array(
        [width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=border_mode, borderValue=(0, 0, 0,))

    mask = mask.astype(np.float32)
    mask = cv2.warpPerspective(mask, mat, (width, height),
                               flags=cv2.INTER_NEAREST,
                               borderMode=border_mode, borderValue=(0, 0, 0,))
    mask = mask.astype(np.int32)

    return image, mask


def do_flip_transpose2(image, masks, type=0):
    # choose one of the 8 cases
    if type == 1:  # rotate90
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

        masks = masks.transpose(1, 0, 2)
        masks = cv2.flip(masks, 1)

    if type == 2:  # rotate180
        image = cv2.flip(image, -1)
        masks = cv2.flip(masks, -1)

    if type == 3:  # rotate270
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 0)

        masks = masks.transpose(1, 0, 2)
        masks = cv2.flip(masks, 0)

    if type == 4:  # flip left-right
        image = cv2.flip(image, 1)
        masks = cv2.flip(masks, 1)

    if type == 5:  # flip up-down
        image = cv2.flip(image, 0)
        masks = cv2.flip(masks, 0)

    if type == 6:
        image = cv2.flip(image, 1)
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

        masks = cv2.flip(masks, 1)
        masks = masks.transpose(1, 0, 2)
        masks = cv2.flip(masks, 1)

    if type == 7:
        image = cv2.flip(image, 0)
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

        masks = cv2.flip(masks, 0)
        masks = masks.transpose(1, 0, 2)
        masks = cv2.flip(masks, 1)

    return image, masks


def do_crop2(image, masks, x, y, w, h):
    H, W = image.shape[:2]

    if H < h:
        h = H
    if W < w:
        w = W

    assert (H >= h)
    assert (W >= w)

    if x == -1 & y == -1:
        x = (W - w) // 2
        y = (H - h) // 2

    if (x, y, w, h) != (0, 0, W, H):
        image = image[y:y + h, x:x + w]
        masks = masks[y:y + h, x:x + w]

    return image, masks


def fix_crop_transform2(image, mask, x, y, w, h):
    return do_crop2(image, mask, x, y, w, h)


def random_crop_transform2(image, masks, w, h, u=0.5):
    x, y = -1, -1
    if random.random() < u:

        H, W = image.shape[:2]
        if H > h:
            y = np.random.choice(H - h)
        else:
            y = 0

        if W > w:
            x = np.random.choice(W - w)
        else:
            x = 0

    return do_crop2(image, masks, x, y, w, h)


# https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
# https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
def do_elastic_transform2(image, mask, grid=32, distort=0.2):
    borderMode = cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=borderMode, borderValue=(0, 0, 0,))

    mask = mask.astype(np.float32)
    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                     borderMode=borderMode, borderValue=(0, 0, 0,))
    mask = mask.astype(np.int32)
    mask = relabel_mask(mask)

    return image, mask
