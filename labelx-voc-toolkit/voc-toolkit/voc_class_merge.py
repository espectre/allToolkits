# -*- coding:utf-8 -*-
from lxml import etree
import os
import sys
# the script to merge multiple class into  one class

merge_config = {
    # key : old class name
    # value : new class name
    "guns_true":"guns",
    "guns_anime":"guns",
    "guns_tools":"guns",
    "knives_true": "knives",
    "knives_false": "knives",
    "knives_kitchen": "knives"
}

def processXmlFun(oldXmlFile=None,newXmlFile=None,merge_config=None):
    tree = etree.parse(oldXmlFile)
    rooTElement = tree.getroot()
    for child in rooTElement:
        if child.tag == "object":
            object_name = child.xpath('name')[0].text
            if object_name in merge_config:
                 child.xpath('name')[0].text = merge_config[object_name]
    tree.write(newXmlFile, pretty_print=True)

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
    anno_9 = annoDir
    anno_5 = os.path.abspath(annoDir)+"-5"
    convertOldToNew(oldAnnoDir=anno_9, newAnnoDir=anno_5, config=merge_config)
    pass

def main():
    pass

if __name__ == '__main__':
    main()
