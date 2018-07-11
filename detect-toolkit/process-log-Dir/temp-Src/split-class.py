# -*- coding:utf-8 -*-
from lxml import etree
import os
import sys
import json


def getClassName(line=None):
    line_dict = json.loads(line)
    key = line_dict['url']
    if line_dict['label'] == None or len(line_dict['label']) == 0:
        return key, None
    label_dict = line_dict['label'][0]
    if label_dict['data'] == None or len(label_dict['data']) == 0:
        return key, None
    data_dict_list = label_dict['data']
    label_bbox_list_elementDict = []
    for bbox in data_dict_list:
        if 'class' not in bbox or bbox['class'] == None or len(bbox['class']) == 0:
            continue
        label_bbox_list_elementDict.append(bbox['class'])
    return label_bbox_list_elementDict


file = "/Users/wangbing/Downloads/0703/20180422/20180422_rfcn.json"

def main():
    class_line_dict =  dict()
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            label_class_name_list = getClassName(line=line)
            for class_name in label_class_name_list:
                if class_name in class_line_dict:
                    class_line_dict[class_name].append(line)
                else:
                    class_line_dict[class_name] = []
                    class_line_dict[class_name].append(line)
    for class_name in class_line_dict:
        class_file_name = file+"-"+class_name.replace(" ",'')+'.json'
        with open(class_file_name,'w') as f:
            f.write('\n'.join(class_line_dict[class_name]))
if __name__ == '__main__':
    main()
