import numpy as np
import os
import re
import pandas as df
import cv2
import src.mask_rcnn.model as modellib
from src.mask_rcnn.bowl_dataset import BowlDataset
from src.mask_rcnn.inference_config import inference_config
from src.utils.utils import get_image_paths, rle_encoding


def predict():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    base_path = "stage2_test_final"

    # Training dataset
    dataset_train = BowlDataset()
    dataset_train.load_bowl(base_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BowlDataset()
    dataset_val.load_bowl(base_path)
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

    # Test all images
    id_extractor = re.compile(".*\\\\" + base_path +
                              "\\\\(?P<image_id>.*)" +
                              "\\\\images\\\\.*")

    out_list = []
    image_paths = get_image_paths(base_path)
    for image_path in image_paths:
        matches = id_extractor.match(image_path)
        image_id = matches.group("image_id")

        final_path = os.path.join("..", base_path, image_id, "final")
        if not os.path.exists(final_path):
            os.mkdir(final_path)

        original_image = cv2.imread(image_path)
        results = model.detect([original_image], verbose=1)

        predicts = results[0]['masks'].copy()

        # 完全去重叠
        sum_predicts = np.sum(predicts, axis=2)
        sum_predicts[sum_predicts >= 2] = 0
        sum_predicts = np.expand_dims(sum_predicts, axis=-1)
        predicts = predicts * sum_predicts

        # sum_predicts = np.sum(predicts, axis=2)
        # rows, cols = np.where(sum_predicts >= 2)
        #
        # for i in zip(rows, cols):
        #     instance_indicies = np.where(np.any(predicts[i[0], i[1], :]))[0]
        #     highest = instance_indicies[0]
        #     predicts[i[0], i[1], :] = predicts[i[0], i[1], :] * 0
        #     predicts[i[0], i[1], highest] = 1

        for i in range(predicts.shape[2]):
            mask = predicts[:, :, i]
            rle = rle_encoding(mask)
            if len(rle) > 0:
                out_list += [dict(ImageId=image_id,
                                  EncodedPixels=" ".join(map(str, rle)))]

    out_path = os.path.join("..", base_path, "predictions.csv")
    out_df = df.DataFrame(out_list)
    out_df[['ImageId', 'EncodedPixels']].to_csv(out_path, index=False)
