# -*- coding:utf-8 -*-
from lxml import etree
import os
import sys
import shutil
import json
import random
config = {
    "not terror": [False,65729],
    "knives_false": [True, 2516],
    "tibetan flag": [True, 2906],
    "islamic flag": [True, 4735],
    "guns_true": [False, 68061],
    "guns_anime": [False, 32996],
    "guns_tools": [True, 1644],
    "knives_true": [False, 16971],
    "knives_kitchen": [False, 8410],
    "isis flag": [True, 3198]
}

def parseXmlFile_countBboxClassNum(xmlFile=None):
    tree = etree.parse(xmlFile)
    rooTElement = tree.getroot()
    object_className_list = []
    for child in rooTElement:
        if child.tag == "object":
            one_object_dict = {}
            one_object_dict['name'] = child.xpath('name')[0].text
            one_object_dict['xmin'] = child.xpath(
                'bndbox')[0].xpath('xmin')[0].text
            one_object_dict['ymin'] = child.xpath(
                'bndbox')[0].xpath('ymin')[0].text
            one_object_dict['xmax'] = child.xpath(
                'bndbox')[0].xpath('xmax')[0].text
            one_object_dict['ymax'] = child.xpath(
                'bndbox')[0].xpath('ymax')[0].text
            object_className_list.append(child.xpath('name')[0].text)
    # 
    return object_className_list


def doubleOrNotLine(line=None):
    xmlFile = os.path.join(vocPath, 'Annotations',line+'.xml')
    classNameList = parseXmlFile_countBboxClassNum(xmlFile=xmlFile)
    doubleFlag = True
    for name in classNameList:
        if config[name][0] == False:
            doubleFlag = False
    return doubleFlag


def doubleListFileFun(file=None):
    imageList = []
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            imageList.append(line)  # orignal image
    print(len(imageList))
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            res = doubleOrNotLine(line=line)
            if res == True:
                imageList.append(line)
    print(len(imageList))
    random.shuffle(imageList)
    with open(file,'w') as f:
        f.write('\n'.join(imageList))

vocPath = ""
def main():
    for i in ['trainval.txt']:
        file = os.path.join(vocPath, 'ImageSets/Main',i)
        shutil.copyfile(file,file+'.original')
        doubleListFileFun(file=file)

    pass


if __name__ == '__main__':
    main()
