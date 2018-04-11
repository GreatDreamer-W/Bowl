import re
from glob import glob

import os

import cv2
import numpy
from src.mask_rcnn import utils


class BowlDataset(utils.Dataset):

    def load_bowl(self, base_paths):
        self.add_class("bowl", 1, "nuclei")
        for base_path in base_paths:
            self.load_part(base_path)

    def load_part(self, base_path):
        masks = dict()
        id_extractor = re.compile(".*\\\\" + base_path +
                                  "\\\\(?P<image_id>.*)" +
                                  "\\\\masks\\\\(?P<mask_id>.*)\.png")

        for mask_path in glob(
                os.path.join("..", base_path, "**", "masks", "*.png")):
            matches = id_extractor.match(mask_path)
            image_id = matches.group("image_id")
            image_paths = glob(os.path.join("..",base_path,
                                            image_id, "images", "*.png"))
            for image_path in image_paths:
                if image_path in masks:
                    masks[image_path].append(mask_path)
                else:
                    masks[image_path] = [mask_path]

        for i, (image_path, mask_paths) in enumerate(masks.items()):
            self.add_image(
                "bowl", image_id=i, path=image_path, mask_paths=mask_paths)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = cv2.imread(info["path"])
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_paths = info["mask_paths"]
        masks = []

        for i, mask_path in enumerate(mask_paths):
            # Load an color image in gray scale
            masks.append(cv2.imread(mask_path, 0))

        masks = numpy.stack(masks, axis=-1)
        masks = numpy.where(masks > 127, 1, 0)

        class_ids = numpy.ones(len(mask_paths), dtype=numpy.int32)
        return masks, class_ids
