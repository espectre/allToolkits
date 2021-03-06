# -*- coding:utf-8 -*-
import os
import sys
import json
import argparse
import labelx2xml_helper

helpInfoStr = \
"""
    labelx convert to pascal voc xml file toolkit   
    --actionFlag : 功能 flag
        1 : 将指定目录下的所有打标过的json 文件转换成 pascal xml 格式数据
            --labelxBasePath ,required
            --vocpath , optional
            --renamePrefix , optional # 用于 rename file 的前缀
        2 : 将一个 pascal voc 数据集 添加到 另外一个数据集中
            --vocpath ,required 
            --finalVocpath , required
            --flag ,"overwrite,append"  # 这个还没有实现
                    overwrite 表示如果 存在 xml文件名相同，那么就覆盖原有的xml文件，重新生成。
                    append 表示如果 存在xml 文件，那么进行 bbox 追加
            将 vocpath 指向的数据集 添加到 finalVocpath 这个数据集中
        3 : 根据已经有的图片和xmL文件生成 ImageSets/Main，readme.txt
            --vocpath ,required 
            --recreateImageSetFlag , optional
                  0:not create ImageSet/Main ,just get readme file , 
                  1: create ImageSet/Main and readme file
        4 : 统计vopath bbox 的类别信息
            --vocpath ,required 
        5 : 抽样画图，抽样画 pascal voc 格式的数据
            --vocpath ,required
            会 将画的图 保存在 vocpath+'-draw' 目录下。
        6 : 对数据集每张图片求 md5 , 也可以删除重复的数据集（由于删除数据的图片，重新生成 trainval test 文件）
            --vocpath,require
            --deleteSameMd5Flag # 0 : not delete ,just get md5 , 1 : delete Sama md5 xml and image
        7 : 对所有的图片，cv2 imread , 然后保存 为 jpg 格式
            --vocpath,require
"""
changeLog = \
"""
    2018-05-19: 
        gen_imagesets : add rename image 
        gen_imagesets : bbox info add to readme.txt
        add utils file ,the file is common used function
    2018-07-10:
        add actionFlag 6 : image md5 check and delete same md5 image
        add actionFlag 7 : cv2.imread image and cv2.imwrite image
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='labelx convert to pascal voc toolkit'
    )
    parser.add_argument('--actionFlag',
                        dest='actionFlag',
                        default=None,
                        required=True,
                        help='action flag int',
                        type=int)
    parser.add_argument('--vocpath',
                        dest='vocpath',
                        default=None,
                        help='vocpath for data generate',
                        type=str)
    parser.add_argument('--finalVocpath',
                        dest='finalVocpath',
                        default=None,
                        help='vocpath for data generate',
                        type=str)
    parser.add_argument('--labelxBasePath',
                        dest='labelxBasePath',
                        default=None,
                        help='labelx annotation file base path',
                        type=str)
    parser.add_argument('--renamePrefix',
                        dest='renamePrefix',
                        default=None,
                        help='rename image : new file prefix',
                        type=str)
    parser.add_argument('--deleteSameMd5Flag',
                        dest='deleteSameMd5Flag',
                        default=0,
                        help='0:just get all md5 , 1: delete same md5 image',
                        type=int)
    parser.add_argument('--recreateImageSetFlag',
                        dest='recreateImageSetFlag',
                        default=0,
                        help='0:not create ImageSet/Main ,just get readme file , 1: create ImageSet/Main and readme file',
                        type=int)
    args = parser.parse_args()
    return args

args = parse_args()

def main():
    if args.actionFlag == None:
        print("WARNING %s" % (helpInfoStr))
    elif args.actionFlag == 1:
        labelxBasePath = args.labelxBasePath
        if labelxBasePath == None:
            print("labelxBasePath required")
            return -1
        labelxBasePath = os.path.abspath(labelxBasePath)
        vocpath = args.vocpath
        if vocpath == None:
            vocpath = labelxBasePath+'-vocResult'
        labelx2xml_helper.covertLabelxMulFilsToVoc_Fun(
            labelxPath=labelxBasePath, vocResultPath=vocpath, renamePrefix=args.renamePrefix)
        pass
    elif args.actionFlag == 2:
        vocpath = args.vocpath
        finalVocpath = args.finalVocpath
        if vocpath == None or finalVocpath == None:
            print("vocpath and finalVocpath is required")
            return -1
        res = labelx2xml_helper.mergePascalDataset(
            littlePath=vocpath, finalPath=finalVocpath)
        if res == 'error':
            return 1
        pass
    elif args.actionFlag == 3:
        vocpath = args.vocpath
        recreateImageSetFlag = args.recreateImageSetFlag
        if vocpath == None:
            print("vocpath is required")
            return -1
        if recreateImageSetFlag == 0:
            labelx2xml_helper.gen_imageset_Fun(vocPath=vocpath,recreateImageSetFlag=False)
        elif recreateImageSetFlag == 1:
            labelx2xml_helper.gen_imageset_Fun(vocPath=vocpath,recreateImageSetFlag=True)
        pass
    elif args.actionFlag == 4:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.statisticBboxInfo_Fun(vocPath=vocpath)
        pass
    elif args.actionFlag == 5:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.drawImageWithBbosFun(vocPath=vocpath)
    elif args.actionFlag == 6:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.getAllImageMD5Fun(
            vocPath=args.vocpath, deleteFlag=args.deleteSameMd5Flag)
    elif args.actionFlag == 7:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.reWriteImageWithCv2(vocPath=vocpath)
    pass

if __name__ == '__main__':
    res = main()
    if res == -1:
        print(helpInfoStr)
    else:
        print("RUN SUCCESS")



