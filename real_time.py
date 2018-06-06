import cv2
import numpy as np
from imutils.video import WebcamVideoStream
import imutils
import os

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

PATH_TO_LABELS = os.path.join('mscoco_label_map.pbtxt')
NUM_CLASSES = 90
PATH_TO_CKPT = os.path.join('ssd_mobilenet_v1_coco_2017_11_17', 'frozen_inference_graph.pb')

# loading a frozen tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

stream = WebcamVideoStream(src=0).start()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            frame = stream.read()
            frame_expanded = np.expand_dims(frame, axis=0)
            frame_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={frame_tensor : frame_expanded}
            )

            vis_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.uint32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5
            )

            cv2.imshow('Object Detection', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
