# -*- coding:utf-8 -*-
import os
import sys
import json
from lxml import etree
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


vocPath = ""
xmlBasePath = os.path.join(vocPath, 'Annotations')
imageBasePath = os.path.join(vocPath, 'JPEGImages')
def main():  
    allImageJustNameNoPostfixFileList = [] 
    for i in os.listdir(imageBasePath):
        imageNoPostfix = i.split('.')[-1]
        allImageJustNameNoPostfixFileList.append(imageNoPostfix)
    deleteImageList = [] # just image name no postfix
    for i in range(len(allImageJustNameNoPostfixFileList)):
        imageName = allImageJustNameNoPostfixFileList[i]
        res = checkXmlSaveOrDelete(imageNameNoPostfix=imageName)
        if res:
            deleteImageList.append(imageName)
    print(len(deleteImageList))


if __name__ == '__main__':
    main()
