# -*- coding:utf -*-
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, "refinenet/python")
import caffe
from argparse import ArgumentParser
import time
import json
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format


def parser_args():
    parser = ArgumentParser('inference script')
    parser.add_argument('--imageFile', help='image list file ',default=None, type=str)
    parser.add_argument('--gpuId', help='gpu id',default=0, type=int)
    parser.add_argument('--bathSize',help='batch size',default=1, type=int)
    # model param
    parser.add_argument('--modelFilePath', required=True, help='path of caffemodel file', type=str)
    parser.add_argument('--deployFilePath', required=True, help='path of deploy file',type=str)
    parser.add_argument('--labelFilePath', required=True, help='label file name',type=str)
    return parser.parse_args()

def change_deploy(deploy_file=None, batch_size=None):
    net = caffe_pb2.NetParameter()
    with open(deploy_file, 'r') as f:
        text_format.Merge(f.read(), net)
    data_layer_index = 0
    data_layer = net.layer[data_layer_index]
    data_layer.input_param.shape[0].dim[0] = batch_size
    with open(deploy_file, 'w') as f:
        f.write(str(net))

def parse_label_file(labelFile=None):
    label_dict = dict()
    with open(labelFile, 'r') as f:
        line_list = [i.strip() for i in f.readlines() if i]
        keyList = line_list[0].split(',')  # index,class,threshold
        for key in keyList[1:]:
            label_dict[key] = dict()
        for i_line in line_list[1:]:
            i_line_list = i_line.split(',')
            index_value = int(i_line_list[0])
            for colume_index, value in enumerate(i_line_list[1:], 1):
                label_dict[keyList[colume_index]][index_value] = value
    return label_dict

def create_net(config):
    caffe.set_mode_gpu()
    caffe.set_device(config['gpuId'])
    net = caffe.Net(config['deployFilePath'], config['modelFilePath'], caffe.TEST)
    labelDict = parse_label_file(config['labelFilePath'])
    return net,labelDict


def postProcessResult(output,batch_image_info,labelDict):
    output_bbox_list = output['detection_out'][0][0]
    image_result_dict = dict()  # image_id : bbox_list
    for i_bbox in output_bbox_list:
        image_id = int(i_bbox[0])
        if image_id >= len(batch_image_info):
            break
        h,w = batch_image_info[image_id][1]
        class_index = int(i_bbox[1])
        if class_index < 1 :
	    continue
        score = float(i_bbox[2])
        if score < float(labelDict['threshold'][class_index]):
            continue
        bbox_dict = dict()
        bbox_dict['index'] = class_index
        bbox_dict['class'] = labelDict['class'][class_index]
        bbox_dict['score'] = score
        bbox = i_bbox[3:7] * np.array([w, h, w, h])
        bbox_dict['pts'] = []
        xmin = int(bbox[0]) if int(bbox[0]) > 0 else 0
        ymin = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xmax = int(bbox[2]) if int(bbox[2]) < w else w
        ymax = int(bbox[3]) if int(bbox[3]) < h else h
        bbox_dict['pts'].append([xmin, ymin])
        bbox_dict['pts'].append([xmax, ymin])
        bbox_dict['pts'].append([xmax, ymax])
        bbox_dict['pts'].append([xmin, ymax])
        if image_id not in image_result_dict.keys():
            image_result_dict[image_id] = []
        image_result_dict[image_id].append(bbox_dict)
    resps = []
    for image_id in range(len(batch_image_info)):
        if image_id in image_result_dict.keys():
            res_list = image_result_dict.get(image_id)
        else:
            res_list = []
        result = {"detections": res_list}
        resps.append(
            {"imageName":batch_image_info[image_id][0], "result": result})
    return resps

def net_inference(net,batchImageList,labelDict):
    def preProcess(imageName):
        image = cv2.imread(imageName)
        img = cv2.resize(image, (512, 512))
        img = img.astype(np.float32, copy=False)
        img = img - np.array([[[103.52, 116.28, 123.675]]])
        # img = img * 0.017
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        return img

    batch_image_info = [] # element : [imageName,[image_height,image_width]
    for index,imageName in enumerate(batchImageList):
        imageData = preProcess(imageName)
        batch_image_info.append([imageName,[imageData.shape[0], imageData.shape[1]]])
        net.blobs['data'].data[index]  = imageData
    output = net.forward()
    res = postProcessResult(output,batch_image_info,labelDict)
    return res
    pass

def init_net(config):
    change_deploy(config['deployFilePath'],config['bathSize'])
    net,labelDict = create_net(config)
    return net,labelDict

def infe_ImageFile(config,net,labelDict):
    with open(config['imageFile'],'r') as f:
        imageList = [i.strip() for i in f.readlines() if i.strip()]
        for i_batch in range(0,len(imageList),config['bathSize']):
            temp_end = i_batch+config['bathSize'] if i_batch+config['bathSize'] < len(imageList) else len(imageList)
            temp_imageList = imageList[i_batch:temp_end]
            res = net_inference(net,temp_imageList,labelDict)
            for i_res in res:
		print(i_res)
                pass
    pass

args = parser_args()
def main():
    config = vars(args)
    print(config)
    net,labelDict = init_net(config)
    infe_ImageFile(config,net,labelDict)
if __name__ == '__main__':
    main()
"""
python infe.py \
--imageFile  \
--gpuId 0 \
--batchSize 1 \
--modelFilePath ../model/wa_C23_v1_vgg16_512x512_iter_24000.caffemodel \
--deployFilePath ../model/deploy.prototxt \
--labelFilePath ../model/labels.csv
"""