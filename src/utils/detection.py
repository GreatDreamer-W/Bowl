import os
import re
from glob import glob

import cv2
from matplotlib import pyplot as plt

import src.mask_rcnn.model as modellib
from src.mask_rcnn.bowl_dataset import BowlDataset
from src.mask_rcnn import visualize
from src.mask_rcnn.inference_config import inference_config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Training dataset
dataset_train = BowlDataset()
dataset_train.load_bowl("stage1_train")
dataset_train.prepare()

# Validation dataset
dataset_val = BowlDataset()
dataset_val.load_bowl("stage1_train")
dataset_val.prepare()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Test all images
base_path = "stage1_test"
masks = dict()
id_extractor = re.compile(".*\\\\" + base_path +
                          "\\\\(?P<image_id>.*)" +
                          "\\\\images\\\\.*")
for image_path in glob(
        os.path.join("..", base_path, "**", "images", "*.png")):
    matches = id_extractor.match(image_path)
    image_id = matches.group("image_id")
    print(image_id)
    mask_path = os.path.join("..", base_path, image_id, "masks")
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    original_image = cv2.imread(image_path)
    results = model.detect([original_image], verbose=1)
    r = results[0]['masks']

    rr = results[0]
    visual = visualize.display_instances(original_image,
                                         rr['rois'],
                                         rr['masks'],
                                         rr['class_ids'],
                                         dataset_val.class_names,
                                         rr['scores'],
                                         ax=get_ax())

    cv2.imwrite(os.path.join(mask_path, "visual.png"), visual)

    # for mask_index in range(r.shape[2]):
    #     mask = r[:, :, mask_index]
    #     for i in range(mask.shape[0]):
    #         for j in range(mask.shape[1]):
    #             if mask[i][j] > 0:
    #                 mask[i][j] = 255
    #     res = cv2.resize(mask, (original_image.shape[1],
    #                             original_image.shape[0]))
    #     cv2.imwrite(os.path.join(mask_path, str(mask_index) + ".png"), res)
