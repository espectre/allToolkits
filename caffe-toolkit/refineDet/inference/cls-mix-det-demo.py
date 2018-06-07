# ------------------------------------------
# AtLab_SSD_Mobilenet_V0.1
# Demo
# by Zhang Xiaoteng
# ------------------------------------------
import numpy as np
import sys
import os
from argparse import ArgumentParser
if not 'caffe/python' in sys.path:
    sys.path.insert(0, 'caffe/python')
import caffe
import time
import cv2
import random


def parser():
    parser = ArgumentParser('AtLab BK-lite Demo!')
    parser.add_argument('--images', dest='im_path', help='Path to the image',
                        default='images', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='The GPU ide to be used',
                        default=0, type=int)
    parser.add_argument('--proto', dest='prototxt', help='BK-lite caffe test prototxt',
                        default='lib/models/deploy-final.prototxt', type=str)
    parser.add_argument('--model', dest='model', help='BK-lite trained caffemodel',
                        default='lib/models/Final_model.caffemodel', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Output path for saving the figure',
                        default='output', type=str)
    return parser.parse_args()


def preprocess(src):
    img = cv2.resize(src, (224, 224))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    img = img.transpose((2, 0, 1))
    return img


def det_postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out[0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out[0, 0, :, 1]
    conf = out[0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)


def cls_postprocess(out):
    score = out[0]
    sort_pre = sorted(enumerate(score), key=lambda z: z[1])
    label_cls = [sort_pre[-j][0] for j in range(1, 2)]
    score_cls = [sort_pre[-j][1] for j in range(1, 2)]
    return label_cls, score_cls


def plot_image(plot_im, num, coco_names, box, conf, cls, image, out_path, cls_names, label_cls, score_cls):
    color_white = (0, 0, 0)
    true_terror_flag = 0
    for i in range(5):
        color = (random.randint(0, 256), random.randint(
            0, 256), random.randint(0, 256))
        for j in range(num):
            if int(cls[j]) == (i+1):
                true_terror_flag = true_terror_flag + 1
                box[j] = map(int, box[j])
                cv2.rectangle(
                    plot_im, (box[j][0], box[j][1]), (box[j][2], box[j][3]), color=color, thickness=3)
                cv2.putText(plot_im, '%s %.3f' % (coco_names[i], float(
                    conf[j])), (box[j][0], box[j][1] + 15), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    if true_terror_flag > 0:
        cv2.imwrite(os.path.join(out_path, image), plot_im)


def detect(net, im_path, det_names, cls_names, image, out_path):

    origimg = cv2.imread(im_path)
    origin_h, origin_w, ch = origimg.shape
    img = preprocess(origimg)

    net.blobs['data'].data[...] = img
    starttime = time.time()
    out = net.forward()
    cls_scores = net.blobs['prob'].data
    det_scores = net.blobs['detection_out'].data
    print np.shape(out)
    print det_scores
    label_cls, score_cls = cls_postprocess(cls_scores)
    print label_cls
    box, conf, cls = det_postprocess(origimg, det_scores)
    num = len(box)
    endtime = time.time()
    per_time = float(endtime - starttime)
    print 'speed: {:.3f}s / iter'.format(endtime - starttime)

    plot_image(origimg, num, det_names, box, conf, cls, image,
               out_path, cls_names, label_cls, score_cls)

    return per_time


if __name__ == "__main__":
    args = parser()
    det_names = np.loadtxt('lib/det.txt', str, delimiter='\n')
    cls_names = np.loadtxt('lib/cls.txt', str, delimiter='\n')
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(
        args.prototxt), 'Please provide a valid path for the prototxt!'
    assert os.path.isfile(
        args.model), 'Please provide a valid path for the caffemodel!'

    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'AtLab-BK-lite'
    print('Done!')

    totle_time = 0
    for image in os.listdir(args.im_path):
        img = os.path.join(args.im_path, image)
        per_time = detect(net, img, det_names, cls_names, image, args.out_path)
        totle_time = totle_time + per_time
    print totle_time
