# -*- coding:utf-8 -*-
from lxml import etree
import os
import sys
import json
# the script to merge multiple class into  one class

merge_config = {
    # key : old class name
    # value : new class name
    "tibetan flag":"tibetan_flag",
    "islamic flag":"islamic_flag",
    "isis flag":"isis_flag",
    "guoqi_flag":"china_guoqi_flag",
    "bairiqi_flag":"taiwan_bairiqi_flag",
    "guns_true":"guns_true",
    "guns_anime":"guns_anime",
    "guns_tools":"guns_tools",
    "knives_true":"knives_true",
    "knives_kitchen": "knives_kitchen",
    "knives_false":"knives_false",
    "nazi": "nazi_logo",
    "falungong":"falungong_logo",
    "falungong_logo":"falungong_logo",
    "mingjinghuopai": "mingjing_logo",
    "zhongguojinwen":"zhongguojinwen_logo",
    "idcard_negative":"idcard_negative",
    "idcard_positive":"idcard_positive",
    "bankcard_positive":"bankcard_positive",
    "bankcard_negative":"bankcard_negative",
    "gongzhang":"gongzhang_logo",
    "not_card":"not_terror_card_text",
    "not terror":"not_terror",
}

def change_readme(readmeFile):
    readmeDict = json.load(open(readmeFile,'r'))
    for key in readmeDict['bboxInfo']: # test or trainval
        for bbox_name in readmeDict['bboxInfo'][key].keys():
            if bbox_name in merge_config:
                new_bbox_name = merge_config[bbox_name]
                readmeDict['bboxInfo'][key][new_bbox_name] = readmeDict['bboxInfo'][key][bbox_name]
                if new_bbox_name != bbox_name:
                    del readmeDict['bboxInfo'][key][bbox_name]
            else:
                print("%s not in merge_config !!!"%(bbox_name))
    with open(readmeFile+"_new",'w') as f:
        json.dump(readmeDict, f, indent=4)


def processXmlFun(oldXmlFile=None,newXmlFile=None,merge_config=None):
    tree = etree.parse(oldXmlFile)
    rooTElement = tree.getroot()
    has_object_flag = False
    for child in rooTElement:
        if child.tag == "object":
            has_object_flag=True
            object_name = child.xpath('name')[0].text
            if object_name in merge_config:
                 child.xpath('name')[0].text = merge_config[object_name]
    tree.write(newXmlFile, pretty_print=True)
    if has_object_flag == False:
        print(newXmlFile+":"+"empty bbox")

def convertOldToNew(oldAnnoDir=None,newAnnoDir=None,config=None):
    if not os.path.exists(newAnnoDir):
        os.makedirs(newAnnoDir)
    for i_annoFile in sorted(os.listdir(oldAnnoDir)):
        oldAnnoDir_xmlFile = os.path.join(oldAnnoDir,i_annoFile)
        newAnnoDir_xmlFile = os.path.join(newAnnoDir,i_annoFile)
        processXmlFun(oldXmlFile=oldAnnoDir_xmlFile,
                      newXmlFile=newAnnoDir_xmlFile, merge_config=config)
    pass

def convert9_to_5(annoDir=None):
    anno_old = annoDir
    anno_new_name = os.path.abspath(annoDir)+"-renameNew"
    convertOldToNew(oldAnnoDir=anno_old, newAnnoDir=anno_new_name, config=merge_config)
    change_readme(os.path.join(annoDir[:annoDir.rfind('/')],'readme.txt'))
    pass

def main():
    annoFilePath = '/workspace/mnt/group/general-reg/wangbingbing/terror-det-dataset/gene_dataset/WA_0102_ADD_dataset/Annotations'
    convert9_to_5(annoDir=annoFilePath)
    pass

if __name__ == '__main__':
    main()