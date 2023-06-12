#!/usr/bin/env python3
"""
Task 0. Process outputs
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Class that executes Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor method of Yolo class

        - model_path: path to where a Darknet Keras model is stored
        - classes_path: path to where the list of class names used for the
        Darknet model, listed in order of index, can be found
        - class_t: float representing the box score threshold for the initial
        filtering step
        - nms_t: float representing the IOU threshold for non-max suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
        all of the anchor boxes:
            - outputs: number of outputs (predictions) made by the Darknet
            model
            - anchor_boxes is the number of anchor boxes used for each
            prediction
            - 2 => [anchor_box_width, anchor_box_height]
        """

        # Set the public attributes:
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as classes:
            self.class_names = [line.strip() for line in classes.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """
        Performs Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Method to process Yolo algorithm outputs

        - outputs: list of numpy.ndarrays containing the predictions from the
        Darknet model for a single image

        Note: Each output will have the shape
        (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            - grid_height & grid_width => the height and width of the grid used
            for the output
            - anchor_boxes => the number of anchor boxes used
            - 4 => (t_x, t_y, t_w, t_h)
            - 1 => box_confidence
            - classes => class probabilities for all classes

        - image_size: numpy.ndarray containing the image's original size
        [image_height, image_width]

        Return:
        a tuple of (boxes, box_confidences, box_class_probs):
            - boxes:  list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
                - 4 => (x1, y1, x2, y2)
                - (x1, y1, x2, y2) should represent the boundary box relative
                to original image
            - box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
            - box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box's class
            probabilities for each output, respectively
        """

        boxes, box_confidences, box_class_probs = [], [], []
        img_h, img_w = image_size
        full_anchor_w = self.anchors[:, :, 0]
        full_anchor_h = self.anchors[:, :, 1]

        # Process each prediction, stored in outputs list:
        for i, predict in enumerate(outputs):
            # 1. Get dimensions of the current prediction:
            grid_h, grid_w, anchor_boxes, _ = predict.shape

            # 2. Get each prediction dimension:
            t_x = predict[:, :, :, 0]
            t_y = predict[:, :, :, 1]
            t_w = predict[:, :, :, 2]
            t_h = predict[:, :, :, 3]

            # 3. Reshape anchor box measures to match grid dimensions:
            anchor_h = np.tile(full_anchor_h[i], grid_h)
            anchor_h = anchor_h.reshape(grid_h, 1, len(full_anchor_h[i]))

            anchor_w = np.tile(full_anchor_w[i], grid_w)
            anchor_w = anchor_w.reshape(grid_w, 1, len(full_anchor_w[i]))

            # 4. Create grid coordinates for positioning boundary boxes:
            gc_x = np.tile(np.arange(grid_w), grid_h).reshape(grid_w,
                                                              grid_w, 1)
            gc_y = np.tile(np.arange(grid_w),
                           grid_h).reshape(grid_h, grid_h).T.reshape(grid_h,
                                                                     grid_h, 1)

            # 5. Boundary boxes: empty array to store processed boundary boxes:
            box = np.zeros(predict[:, :, :, :4].shape)

            # 6. Boundary boxes: get the absolute coordinates:
            b_x = (self.sigmoid(-t_x) + gc_x) / grid_w
            b_y = (self.sigmoid(-t_y) + gc_y) / grid_h

            # 7. Boundary boxes: get the dimensions:
            b_h = anchor_h * np.exp(t_h) / self.model.input.shape[1].value
            b_w = anchor_w * np.exp(t_w) / self.model.input.shape[2].value

            # 8. Store boxes coordinates and dimensions:
            # top-left coordinate: represents the upper-left corner of the
            # bounding box:
            x1 = (b_x - (b_w / 2)) * img_w
            y1 = (b_y - (b_h / 2)) * img_h

            # bottom-right coordinate: represents the lower-right corner of the
            # bounding box:
            x2 = (b_x + (b_w / 2)) * img_w
            y2 = (b_y + (b_h / 2)) * img_h

            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            # 9. Get the box confidences for each output:
            ax = self.sigmoid(predict[:, :, :, 4])
            box_confidences.append(ax.reshape(grid_h, grid_w, anchor_boxes, 1))

            # 10. Calculate the box class probabilities:
            ax = predict[:, :, :, 5:]
            box_class_probs.append(self.sigmoid(ax))

        return boxes, box_confidences, box_class_probs
