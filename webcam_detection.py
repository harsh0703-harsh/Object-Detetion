import numpy as np
import tarfile 
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
import os
import cv2

cap=cv2.VideoCapture(0)

MODEL_NAME = 'object_detection/ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_complete_label_map.pbtxt')



NUM_CLASSES=4

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret,image_np=cap.read()
            image_np_expanded=np.expand_dims(image_np,axis=0)
            image_tensors=detection_graph.get_tensor_by_name('image_tensor:0')
            boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
            scores=detection_graph.get_tensor_by_name('detection_scores:0')
            classes=detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections=detection_graph.get_tensor_by_name('num_detections:0')
            (boxes,scores,classes,num_detections)=sess.run([boxes,scores,classes,num_detections],feed_dict={image_tensors:image_np_expanded})

            vis_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow("Fruits",image_np)
            if cv2.waitKey(1) & 0xFF==ord("q"):
                cv2.destroyAllWindows()
                break
