import os
import yaml
import json
import pprint
from argparse import ArgumentParser


def parser_args():
    parser = ArgumentParser('inference :::::')
    parser.add_argument('-y', '--yamlFile', dest='yamlFile',
                        help='set yaml config file path', default=None, type=str, required=True)
    parser.add_argument('-i','--inputFile', dest='inputFile', help='image list file : url or local imgage abs path',
                        default=None, type=str, required=True)
    # inputFileFlag : the inputFile format
    #                  0 local image absolute path list file
    #                  1  url list file
    parser.add_argument('-f','--inputFileFlag', dest='inputFileFlag', help='0:local image file absolute path, 1:url',
                        default=0, type=int)
    # outputFileFlag : 0 model output direct write to file,
    #                  1 regression format file，
    #                  2 labelx json list format file
    parser.add_argument('-o','--outputFileFlag', dest='outputFileFlag',help='output file flag', type=int,default=0)
    parser.add_argument('-g','--gpuId', dest='gpuId', help='gpuId',default=0, type=int)
    return parser.parse_args()
def processYamlFun(yamlFile=None):
    global CONFIG
    with open(yamlFile,'r') as f:
        CONFIG = yaml.load(f)














args = parser_args()
CONFIG = None
def main():
    pprint.pprint(args)
    processYamlFun(yamlFile=args.yamlFile)
    pprint.pprint(CONFIG)

if __name__ == '__main__':
    main()
