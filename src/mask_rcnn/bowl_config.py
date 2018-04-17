from src.mask_rcnn.config import Config
import numpy as np


class BowlConfig(Config):
    """Configuration for training on the Bowl dataset.
    Derives from the base Config class and overrides values specific to the Bowl
     dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bowl"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 400

    STEPS_PER_EPOCH = 600

    VALIDATION_STEPS = 50

    BACKBONE = "resnet101"

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9

    # # Image mean (RGB)
    # MEAN_PIXEL = np.array([0, 0, 0])


bowl_config = BowlConfig()
bowl_config.display()
