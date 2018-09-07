# -*- coding:utf-8 -*-
import os
import sys
import argparse
import mxnet as mx
import numpy as np
import cv2

curr_path = os.path.abspath(os.path.dirname(__file__))


def get_image(filename):
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (channel, height, width)
    img = img[np.newaxis, :]  # extend to (example, channel, heigth, width)
    return img


def get_label():
    class_label_list = []
    with open(os.path.join(curr_path, '../data/sysnets.txt')) as f:
        for line in f:
            class_label_list.append(line.split(',')[0])
    return class_label_list
    pass


CLASS_LABEL = get_label()


def parse_args():
    parser = argparse.ArgumentParser(
        description="predict data with pretrained resnet-50")
    parser.add_argument('--modelPrefix', dest='modelPrefix')
    parser.add_argument('--epochFlag', dest='epochFlag')
    parser.add_argument('--imagePath', dest='imagePath')
    parser.add_argument('--testDataPath', dest='testDataPath')
    parser.add_argument('--saveDataPath', dest='saveDataPath')
    parser.add_argument('--gpuId', dest='gpuId')
    return parser.parse_args()


def checkFileIsImags(filePath):
    if ('JPEG' in filePath.upper()) or ('JPG' in filePath.upper()) \
            or ('PNG' in filePath.upper()) or ('BMP' in filePath.upper()):
        return True
    return False
    pass


def getImagePathList(imagePath=None):
    allImageList = []
    for parent, dirnames, filenames in os.walk(imagePath):
        for f in filenames:
            if checkFileIsImags(os.path.join(parent, f)):
                allImageList.append(os.path.join(parent, f))
    return allImageList
    pass


def copyPredictImageToSavePath(label_index=None, imagePath=None, saveBasePath=None):
    saveBasePath = os.path.join(saveBasePath, CLASS_LABEL[label_index])
    if os.path.exists(saveBasePath) == False:
        os.makedirs(saveBasePath)
    if "not terror" in imagePath:
        imagePath = imagePath.split(" ")[0]+"\ "+imagePath.split(" ")[1]
    cmd = "cp %s %s" % (imagePath, saveBasePath)
    os.system(cmd)
    pass


def main():
    args = parse_args()
    batch_size = 16
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(
        curr_path, "../model/"+args.modelPrefix), int(args.epochFlag))
    mod = mx.mod.Module(symbol=sym, label_names=None,
                        context=mx.gpu(int(args.gpuId)))
    mod.bind(for_training=False, data_shapes=[
             ('data', (batch_size, 3, 224, 224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    allImageList = getImagePathList(args.testDataPath)
    for i in range(0, len(allImageList)/batch_size):
        idx = list(range(i*batch_size, (i+1)*batch_size))
        img = np.concatenate([get_image(allImageList[i_path])
                              for i_path in idx])
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])
        mod.forward(Batch([mx.nd.array(img)]))
        prob = mod.get_outputs()[0].asnumpy()
        pred = np.argsort(prob, axis=1)
        top1 = pred[:, -1]
        for top1_index, top1_element in enumerate(top1):
            copyPredictImageToSavePath(
                label_index=top1_element, imagePath=allImageList[idx[top1_index]], saveBasePath=args.saveDataPath)
        pass
    pass


if __name__ == '__main__':
    main()

"""
python /workspace/data/bk_dao-qiang-cls_DIR/src/new_predict-dao-qiang.py --modelPrefix pretrain_resnet_50 --epochFlag 20  --testDataPath /workspace/data/dao_qiang_data/terror-det-result-output/tempData --saveDataPath /workspace/data/dao_qiang_data/terror-det-result-output/tempSavePath --gpuId 0
"""
