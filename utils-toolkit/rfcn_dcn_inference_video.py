#-*- coding:utf-8 -*-

import _init_paths

import argparse
import os
import sys
import time
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/rfcn/cfgs/rfcn_coco_demo.yaml')

#sys.path.insert(0, os.path.join(
#    cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from random import random as rand


def parse_args():
    parser = argparse.ArgumentParser(
        description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)',
                        default=False, action='store_true')
    parser.add_argument('--videoFile', help='video file path', type=str)
    # 设置 间隔 ，没间隔 n 帧处理一次，默认是每帧都处理
    parser.add_argument('--interval', dest='interval',
                        default=1, required=False, help='', type=int)
    args = parser.parse_args()
    return args
args = parse_args()


class Rfcn_dcn_model:
    def __init__(self):
        pprint.pprint(config)
        config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
        sym_instance = eval(config.symbol + '.' + config.symbol)()
        self.sym = sym_instance.get_symbol(config, is_train=False)
        self.arg_params, self.aux_params = load_param(cur_path + '/../model/' + (
            'rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'), 0, process=True)
        # set up class names
        self.num_classes = 81
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        #self.need_classes = ['person', 'car', 'motorbike','bus','truck']
        self.need_classes = ['car', 'motorbike', 'bus', 'truck']
        self.colors = {'car': (rand(), rand(), rand()),
                       'motorbike': (rand(), rand(), rand()),
                       'bus': (rand(), rand(), rand()),
                       'truck': (rand(), rand(), rand())}
        #self.warm_up()
        pass

    def load_data_and_get_predictor(self, image_names):
        # load demo data

        #image_names = ['COCO_test2015_000000000891.jpg',
        #            'COCO_test2015_000000001669.jpg']
        data = []
        for im_name in image_names:
            #assert os.path.exists(
            #    cur_path + '/../demo/' + im_name), ('%s does not exist'.format('../demo/' + im_name))
            #im = cv2.imread(cur_path + '/../demo/' + im_name,
            #                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            im = cv2.imread(im_name, cv2.IMREAD_COLOR |
                            cv2.IMREAD_IGNORE_ORIENTATION)
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size,
                                  stride=config.network.IMAGE_STRIDE)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array(
                [[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': im_tensor, 'im_info': im_info})

        # get predictor
        self.data_names = ['data', 'im_info']
        label_names = []
        data = [[mx.nd.array(data[i][name]) for name in self.data_names]
                for i in xrange(len(data))]
        max_data_shape = [[('data', (1, 3, max(
            [v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        provide_data = [[(k, v.shape) for k, v in zip(self.data_names, data[i])]
                        for i in xrange(len(data))]
        provide_label = [None for i in xrange(len(data))]
        self.predictor = Predictor(self.sym, self.data_names, label_names,
                                   context=[mx.gpu(1)], max_data_shapes=max_data_shape,
                                   provide_data=provide_data, provide_label=provide_label,
                                   arg_params=self.arg_params, aux_params=self.aux_params)
        self.nms = gpu_nms_wrapper(config.TEST.NMS, 0)

        return data

    def warm_up(self):
        image_names = ['COCO_test2015_000000000891.jpg',
                       'COCO_test2015_000000001669.jpg']
        data = self.load_data_and_get_predictor(image_names)
        # warm up
        for j in xrange(2):
            data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                         provide_data=[
                [(k, v.shape) for k, v in zip(self.data_names, data[0])]],
                provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2]
                      for i in xrange(len(data_batch.data))]
            scores, boxes, data_dict = im_detect(
                self.predictor, data_batch, self.data_names, scales, config)

    def show_save_boxes(self, im, dets, classes, im_name, scale=1.0):
        #fig = plt.figure()
        dpi = 80
        height, width, nbands = im.shape
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plt.cla()
        plt.axis("off")
        plt.imshow(im)
        for cls_idx, cls_name in enumerate(classes):
            if cls_name not in self.need_classes:
                continue
            cls_dets = dets[cls_idx]
            for det in cls_dets:
                bbox = det[:4] * scale
                #color = (rand(), rand(), rand())
                # 将car 以及truck合并
                if cls_name == 'truck':
                    cls_name = 'car'
                color = self.colors[cls_name]
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=2)
                plt.gca().add_patch(rect)

                if cls_dets.shape[1] == 5:
                    score = det[-1]
                    plt.gca().text(bbox[0], bbox[1],
                                   '{:s} {:.3f}'.format(cls_name, score),
                                   bbox=dict(facecolor=color, alpha=0.2),
                                   fontsize=8, color='white')
        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        plt.show()
        im_name = os.path.splitext(im_name)[0]
        #plt.savefig(im_name + '_bbox.png')
        fig.savefig(im_name + '_bbox.jpg', dpi=dpi, transparent=True)
        return im

    def inference(self, image_names):
        # get data
        data = self.load_data_and_get_predictor(image_names)
        # test
        for idx, im_name in enumerate(image_names):
            data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[
                [(k, v.shape) for k, v in zip(self.data_names, data[idx])]],
                provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2]
                      for i in xrange(len(data_batch.data))]

            tic()
            scores, boxes, data_dict = im_detect(
                self.predictor, data_batch, self.data_names, scales, config)
            boxes = boxes[0].astype('f')
            scores = scores[0].astype('f')
            dets_nms = []
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:,
                                  4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = self.nms(cls_dets)
                cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > 0.6, :]
                dets_nms.append(cls_dets)
            print 'testing {} {:.4f}s'.format(im_name, toc())
            # visualize
            #im = cv2.imread(cur_path + '/../demo/' + im_name)
            im = cv2.imread(im_name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.show_save_boxes(im, dets_nms, self.classes, im_name, 1)
            #show_boxes(im, dets_nms, self.classes, 1)
        pass


def main():
    rfcn_dcn_model = Rfcn_dcn_model()
    #image_names = ['COCO_test2015_000000000891.jpg',
    #               'COCO_test2015_000000001669.jpg']
    images_paths = read_images(args.img_folder)
    print(images_paths)
    rfcn_dcn_model.inference(images_paths)
    pass


def get_write_video()


def main():
    print("*"*10+" runging %s " % (args.videoFile) + '*'*10)
    videoFile = args.videoFile
    frame_inter = args.interval
    time_str = time.strftime("%Y-%m-%d-%H", time.localtime())
    saveJsonResultFile = args.videoFile + '-'+time_str+'-result.json'
    savefileop = open(saveJsonResultFile, 'a+')
    visualizePath = '/workspace/inference/result/vis_result'
    cap = cv2.VideoCapture(videoFile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    print(fourcc)
    write_fps = fps
    videoWriter = cv2.VideoWriter(
        '/workspace/inference/result/bkResult.avi',  fourcc, write_fps, size)

    print('cap is open', cap.isOpened())
    count = 0
    frame_infer = 0
    model_params_list = init_detect_model()
    while True:
        ret, image = cap.read()
        if(ret == False):
            break
        if(count == frame_infer):

            im = process_image_fun(
                imagesPath=image, fileOp=savefileop, vis=visualizePath, model_params_list=model_params_list, count=count)
        #print(im)

            #cv2.waitKey(2)

            videoWriter.write(im)
            frame_infer = frame_infer + frame_inter

        # cv2.imshow('cap video', frame)
        count = count + 1
        if(count % 50 == 0):
            print("Now process the %d image:" % count)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()     # close all the widows opened inside the program
    cap.release()        # release the video read/write handler
    videoWriter.release()

if __name__ == '__main__':
    main()
