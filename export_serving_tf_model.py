import numpy as np
import cv2
import os, shutil
import tensorflow as tf
from lib.model.nms_cpu_fast import non_max_suppression_fast as nms
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

CLASSES = ('__background__','1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = os.path.join('output','frozen_inference_graph.pb')
        self.PATH_TO_LABELS = 'roadsign_label_map.pbtxt'
        self.NUM_CLASSES = 10
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, all_image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                for image_path in all_image:
                    print('detecting: ....{}'.format(image_path))

                    image = cv2.imread(image_path)
                    originalshape= image.shape

                    image = cv2.resize(image, (20, 70), interpolation=cv2.INTER_CUBIC)
                    ratio = [originalshape[0]/20, originalshape[1]/70]
                    print("ratio",ratio)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # print('Detection took {:.3f}s for {:d} object proposals'.format( timer.total_time,boxes.shape[0]))

                    print(boxes[0].shape)
                    print(scores.shape)
                    print(classes.shape)
                    print(num_detections)

                    # dets = np.hstack((boxes[0],
                    #                   scores.transpose())).astype(np.float32)
                    # print('dets.shape',dets)
                    # keep = nms(dets, 0.3)
                    # print(keep)
                    # nms_boxes = dets[keep,:4]
                    # nms_scores = dets[keep,:-1]
                    # nms_clsses = classes[0][keep]



                    nms_boxes = boxes[0]
                    nms_scores = scores[0]
                    nms_clsses = classes[0]

                    nms_boxes, nms_scores, nms_clsses = after_nms(image, nms_boxes, nms_scores, nms_clsses)
                    print(nms_boxes)
                    print(nms_boxes.shape, nms_scores.shape, nms_clsses.shape)
                    fig, ax = plt.subplots(figsize=(12, 12))
                    ax.imshow(image, aspect='equal')
                    image, clc_name = vis_detections_plt2(image, nms_boxes, nms_scores, nms_clsses, ax)
                    sorted_name = sorted(clc_name.items(), key=lambda d: d[0])
                    # print(sorted_name)
                    name = [x[-1] for x in sorted_name]
                    print(name)
                    ax.set_title((name),
                                 fontsize=14)
                    plt.show()
                    # print('test_out/'+image_path.split('/')[-1])

    def serving(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                export_path = 'tf_serving_export_{}_{}'.format('mobilenet', 'pascal')
                print('Exporting trained model to', export_path)
                if os.path.exists(export_path):
                    shutil.rmtree(export_path)
                builder = saved_model_builder.SavedModelBuilder(export_path)

                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_tensor_info = utils.build_tensor_info(image_tensor)
                boxes_tensor_info = utils.build_tensor_info(boxes)
                scores_tensor_info = utils.build_tensor_info(scores)
                classes_tensor_info = utils.build_tensor_info(classes)
                num_detections_tensor_info = utils.build_tensor_info(num_detections)

                prediction_signature = signature_def_utils.build_signature_def(
                    inputs={'image_tensor': image_tensor_info},
                    outputs={'boxes': boxes_tensor_info,
                             'scores': scores_tensor_info,
                             'classes': classes_tensor_info,
                             'num_detections': num_detections_tensor_info,
                             },
                    method_name=signature_constants.PREDICT_METHOD_NAME)
                legacy_init_op = tf.group(tf.initialize_all_tables(),
                                          name='legacy_init_op')
                builder.add_meta_graph_and_variables(
                    sess, [tag_constants.SERVING],
                    signature_def_map={
                        'predict_bbox':
                            prediction_signature,
                    },
                    legacy_init_op=legacy_init_op)

                builder.save()

                print('Done exporting!')
                # python ./tools/export_tf_serving.py
                # tensorflow_model_server --port=9000 --model_name=fasterrcnn --model_base_path=/tmp/tf_serving_export_res101_pascal_voc/
                # python ./tools/tf_faster_rcnn_client.py  --server=localhost:9000

    def loadckpt(self):
        tf.reset_default_graph()
        MODEL_DIR = os.path.abspath(os.path.dirname(__file__)) + "/output"
        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
        last_model = checkpoint.model_checkpoint_path
        print("load {}".format(last_model))
        saver = tf.train.import_meta_graph(last_model + '.meta', clear_devices=CLEAR_DEVICES)
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()


        with tf.Session() as sess:
            saver.restore(sess, last_model)
            export_path = 'tf_serving_export_{}_{}'.format('mobilenet', 'pascal')
            print('Exporting trained model to', export_path)
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            builder = saved_model_builder.SavedModelBuilder(export_path)

            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            image_tensor_info = utils.build_tensor_info(image_tensor)
            boxes_tensor_info = utils.build_tensor_info(boxes)
            scores_tensor_info = utils.build_tensor_info(scores)
            classes_tensor_info = utils.build_tensor_info(classes)
            num_detections_tensor_info = utils.build_tensor_info(num_detections)

            prediction_signature = signature_def_utils.build_signature_def(
                inputs={'image_tensor': image_tensor_info},
                outputs={'boxes': boxes_tensor_info,
                         'scores': scores_tensor_info,
                         'classes': classes_tensor_info,
                         'num_detections': num_detections_tensor_info,
                         },
                method_name=signature_constants.PREDICT_METHOD_NAME)
            legacy_init_op = tf.group(tf.initialize_all_tables(),
                                      name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                signature_def_map={
                    'predict_bbox':
                        prediction_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()

            print('Done exporting!')
            # python ./tools/export_tf_serving.py
            # tensorflow_model_server --port=9000 --model_name=fasterrcnn --model_base_path=/tmp/tf_serving_export_res101_pascal_voc/
            # python ./tools/tf_faster_rcnn_client.py  --server=localhost:9000

    def load_images(self, images_DIR):
      all_inps = os.listdir(images_DIR)  # 读取路径下所有图片
      all_path=[]
      for index in all_inps:
        all_path.append(os.path.join(images_DIR, index))
      print('all_path: {}'.format(all_path))
      return all_path

def plot_sum_best_box(im, image_name, bbest_box, bbest_score, best_ind, ax):

    ax.imshow(im, aspect='equal')

    bbox = bbest_box
    score = bbest_score

    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
    )
    ax.text(bbox[0], bbox[1] - 2,
            '{:s} {:.3f}'.format(CLASSES[best_ind], bbest_score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')

    ax.set_title(('{} detections').format(image_name), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_detections_cv2(resim, bbox, scores, class_ind, thresh=0.5):
    """show detected bounding boxes."""
    inds = np.where(scores >= thresh)[0]
    print('ind:{}'.format(inds))
    # print(bbox)
    # print(scores)
    # print(class_ind)
    [wid, hei, slid] = resim.shape
    color = [0,0,255]
    for i in inds:
        print('i:{}'.format(i))
        bbox_tmp = bbox[i]
        score = scores[i]
        print('bbox_tmp:{}'.format(bbox_tmp))
        sy1 = bbox_tmp[0] * hei
        sx1 = bbox_tmp[1] * wid
        sy2 = bbox_tmp[2] * hei
        sx2 = bbox_tmp[3] * wid
        boxtext = '{:s} {:.3f}'.format(CLASSES[int(class_ind[i])], score)
        print(boxtext, sx1, sy1, sx2, sy2)

        cv2.rectangle(resim, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color, 3)
        cv2.putText(resim, boxtext, (int(sx1), int(sy1 - 3)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color)
    return resim

def after_nms(resim, bbox, scores, class_ind, thresh=0.2):
    [wid, hei, slid] = resim.shape
    inds = np.where(scores>=thresh)[0]
    sy1 = bbox[inds,0] * wid
    sx1 = bbox[inds,1] * hei
    sy2 = bbox[inds,2] * wid
    sx2 = bbox[inds,3] * hei
    sbbox = bbox[inds,:]
    sscores = scores[inds]
    sclass_ind = class_ind[inds]
    print(sy1,sx1,sy2,sx2,sscores,sclass_ind)
    idxs = np.argsort(sy1)
    result_bbox, result_scores, result_classind = [],[],[]
    if len(idxs)>2:
        for index, i in enumerate(idxs[:-1]):
            xx1 = np.maximum(sx1[i], sx1[idxs[index+1]])
            yy1 = np.maximum(sy1[i], sy1[idxs[index+1]])
            xx2 = np.minimum(sx2[i], sx2[idxs[index+1]])
            yy2 = np.minimum(sy2[i], sy2[idxs[index+1]])
            print('xxyy:',xx1,yy1,xx2,yy2)
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 +1)
            h = np.maximum(0, yy2 - yy1 +1)
            overlap = (w * h) /(abs((sx2[i]-sx1[i])*(sy2[i]-sy1[i])) + abs((sx2[idxs[index+1]]-sx1[idxs[index+1]])*(sy2[idxs[index+1]]-sy1[idxs[index+1]])) - w*h)
            print(overlap, w*h, abs((sx2[i]-sx1[i])*(sy2[i]-sy1[i])), abs((sx2[idxs[index+1]]-sx1[idxs[index+1]])*(sy2[idxs[index+1]]-sy1[idxs[index+1]])))
            if overlap>=thresh:
                loca = np.argmax([sscores[i],sscores[int(idxs[index+1])]])
                print('i: {}, i+1:{},  inds: {}, location_min:{}, overlap:{}, wh:{}'.format(i ,int(idxs[index+1]), idxs, loca, overlap, w*h))
                result_bbox.append([sy1[idxs[index+loca]],sx1[idxs[index+loca]],sy2[idxs[index+loca]],sx2[idxs[index+loca]]])
                result_scores.append(sscores[idxs[index+loca]])
                result_classind.append(sclass_ind[idxs[index+loca]])
                # print(type(sbbox))
                # sbbox = np.delete(sbbox, idxs[index+loca], 0)
                # sscores = np.delete(sscores, idxs[index+loca], 0)
                # sclass_ind = np.delete(sclass_ind, idxs[index + loca], 0)
                # print(sbbox)
            else:
                result_bbox.append([sy1[i],sx1[i],sy2[i],sx2[i]])
                result_scores.append(sscores[i])
                result_classind.append(sclass_ind[i])
                if index == len(idxs) - 2:
                    result_bbox.append([sy1[idxs[index + 1]], sx1[idxs[index + 1]], sy2[idxs[index + 1]],
                                        sx2[idxs[index + 1]]])
                    result_scores.append(sscores[idxs[index + 1]])
                    result_classind.append(sclass_ind[idxs[index + 1]])
                    print('add.........')
    else:
        result_bbox = np.array([bbox[inds,0] * wid, bbox[inds,1] * hei, bbox[inds,2] * wid, bbox[inds,3] * hei])
        result_bbox = np.transpose(result_bbox)
        result_scores = sscores
        result_classind = sclass_ind
    return np.array(result_bbox),np.array(result_scores), np.array(result_classind)
    # return sbbox, sscores, sclass_ind

def vis_detections_plt2(resim, bbox, scores, class_ind, ax,thresh=0.6):
    """show detected bounding boxes."""


    inds = np.where(scores >= thresh)[0]
    print('ind:{}'.format(inds))
    # print(bbox)
    # print(scores)
    # print(class_ind)
    [wid, hei, slid] = resim.shape
    clc_name={}
    for i in inds:
        print('ind:{}, score:{}, box:{}'.format(inds[i], scores[i], bbox[i]))
        bbox_tmp = bbox[i]
        score = scores[i]
        # print('bbox_tmp:{}'.format(bbox_tmp))
        sy1 = bbox_tmp[0]
        sx1 = bbox_tmp[1]
        sy2 = bbox_tmp[2]
        sx2 = bbox_tmp[3]
        class_name = CLASSES[int(class_ind[i])]
        print(class_name, sx1, sy1, sx2, sy2)

        ax.add_patch(
            plt.Rectangle((int(sx1), int(sy1)),
                          int(sx2) - int(sx1),
                          int(sy2) - int(sy1), fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(int(sx1),  int(sy1) - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        clc_name[-sx1] = class_name
    ax.set_title(('{} detections with thresh >= {:.1f}').format(clc_name, thresh))

    return resim, clc_name

if __name__ == '__main__':
    # txtpath = 'PascalVOC/ImageSets/Main.txt'
    # imgpath = ['PascalVOC/JPEGImages_temp/2-9.jpg','/']
    # image = cv2.imread(imgpath)
    detector = TOD()
    # all_image = detecotr.load_images('test/')
    # detecotr.detect(all_image)
    detector.serving()

