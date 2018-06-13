# -*- coding:utf-8 -*-
from lxml import etree
import os
import sys

def processXmlFun(xmlFile=None):
    tree = etree.parse(xmlFile)
    rooTElement = tree.getroot()
    for child in rooTElement:
        if child.tag == "filename":
            newFileName = child.text.split('/')[-1]
            child.text = newFileName
    tree.write(xmlFile, pretty_print=True)


def convertOldToNew(annoDir=None):
    for i_annoFile in sorted(os.listdir(annoDir)):
        xmlfile = os.path.join(annoDir,i_annoFile)
        processXmlFun(xmlFile=xmlfile)
    pass


def convert9_to_5(annoDir=None):
    convertOldToNew(annoDir=annoDir)
    pass


def main():
    pass


if __name__ == '__main__':
    main()
