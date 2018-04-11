import os

from src.mask_rcnn import model as modellib

from src.mask_rcnn.bowl_config import bowl_config

from src.mask_rcnn.bowl_dataset import BowlDataset


def train(paths):
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn", "mask_rcnn_coco.h5")

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=bowl_config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "last"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Training dataset
    dataset_train = BowlDataset()
    dataset_train.load_bowl(paths)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BowlDataset()
    dataset_val.load_bowl(paths)
    dataset_val.prepare()

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE,
                epochs=30,
                layers="all")
