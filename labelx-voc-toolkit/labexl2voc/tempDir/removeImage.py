# -*- coding:utf-8 -*-
import os
import sys
import json
from lxml import etree
import random
"""
这个函数的作用是 ：去除一些不想要的图片
"""


def get_object_list(xmlFile=None):
    tree = etree.parse(xmlFile)
    rooTElement = tree.getroot()
    object_list = []
    for child in rooTElement:
        if child.tag == "object":
            bbox_label = child.xpath('name')[0].text
            xmin = child.xpath('bndbox')[0].xpath('xmin')[0].text
            ymin = child.xpath('bndbox')[0].xpath('ymin')[0].text
            xmax = child.xpath('bndbox')[0].xpath('xmax')[0].text
            ymax = child.xpath('bndbox')[0].xpath('ymax')[0].text
            bbox_position = [float(xmin), float(
                ymin), float(xmax), float(ymax)]
            bbox_position = [int(i) for i in bbox_position]
            object_list.append([bbox_label, bbox_position])
    return object_list


def checkBBoxList(bboxList=None,config=None):
    allBboxLenght = len(bboxList)
    tempCount = 0
    for i_bbox in bboxList:
        class_name = i_bbox[0]
        if class_name == 'guns_true':
            tempCount += 1
    if tempCount == allBboxLenght:
        return True
    else:
        return False


def checkXmlSaveOrDelete(imageNameNoPostfix=None,deleteFlag=False):
    xmlFile = os.path.join(xmlBasePath,imageNameNoPostfix+'.xml')
    imageName = os.path.join(imageBasePath,imageNameNoPostfix+'.jpg')
    object_list = get_object_list(xmlFile=xmlFile)
    if len(object_list) == 0:
        print("ERROR %s"%(imageNameNoPostfix))
        exit()
    res = checkBBoxList(bboxList=object_list)
    return res
    pass


def deleteImageFun(deleteImageList=None):
    vocAnno = os.path.join(saveOnlyGunsTrueVocPath, 'Annotations')
    vocJpeg = os.path.join(saveOnlyGunsTrueVocPath, 'JPEGImages')
    if not os.path.exists(vocAnno):
        os.makedirs(vocAnno)
    if not os.path.exists(vocJpeg):
        os.makedirs(vocJpeg)
    for index,i in enumerate(deleteImageList):
        if index % 1000 == 0:
            print("processing : %d"%(index))
        xml = os.path.join(xmlBasePath,i+'.xml')
        newXml = os.path.join(vocAnno, i+'.xml')
        image = os.path.join(imageBasePath,i+'.jpg')
        newImage = os.path.join(vocJpeg, i+'.jpg')
        cmdStr = "mv %s %s"%(xml,newXml)
        os.system(cmdStr)
        cmdStr = "mv %s %s"%(image,newImage)
        os.system(cmdStr)
        pass
    pass

vocPath = ""
saveOnlyGunsTrueVocPath = "/workspace/data/BK/terror-dataSet-Dir/generate_V1.1.2/TERROR-DETECT-temp-0712-only-guns-true"
xmlBasePath = os.path.join(vocPath, 'Annotations')
imageBasePath = os.path.join(vocPath, 'JPEGImages')
def main():  
    allImageJustNameNoPostfixFileList = [] 
    for i in os.listdir(imageBasePath):
        imageNoPostfix = i.split('.')[0]
        allImageJustNameNoPostfixFileList.append(imageNoPostfix)
    deleteImageList = [] # just image name no postfix
    for i in range(len(allImageJustNameNoPostfixFileList)):
        imageName = allImageJustNameNoPostfixFileList[i]
        res = checkXmlSaveOrDelete(imageNameNoPostfix=imageName)
        if res:
            deleteImageList.append(imageName)
    print("Image length : %d"%(len(deleteImageList)))
    get_8000_only_guns_true_list = random.sample(
        deleteImageList, 8000)
    deleteImageFun(deleteImageList=get_8000_only_guns_true_list)



if __name__ == '__main__':
    main()
