#coding=utf-8
"""
    python 3
"""
from urllib.parse import parse_qs as parse_qs
import base64
import json
import re
import yaml
from shutil import copyfile
from downloadFile import *
from image_help import downloadImage_By_urllist
RES_FILE = None
IMAGE_PREFIX = None
IMAGE_COUNT = None
CONFIG = None


def parseImage(raw_query=None):
    req = parse_qs(raw_query)
    url = req['url'][0]
    url = base64.urlsafe_b64decode(url).decode('utf-8')
    if len(url.split('?')) > 2:
        raise Exception("url error : %s" % (url))
    url = url[:url.find('?')]
    return url


def parseResbody(body=None):
    """
        body is string
    """
    res_dict = dict()
    body = json.loads(body)
    if 'code' not in body or body['code'] != 0 or 'fileList' not in body:
        return None
    label = body['fileList'][0]['label']
    rate = body['fileList'][0]['rate']
    review = body['fileList'][0]['review']
    res_dict['label'] = label
    res_dict['rate'] = rate
    res_dict['review'] = review
    if label == 2 and 'objects' in body['fileList'][0]:
        objects = body['fileList'][0]['objects']
        if len(objects) > 0:
            res_dict['objects'] = objects[0]
    return res_dict


def parseLine(line=None):
    global RES_FILE
    global IMAGE_COUNT
    global IMAGE_PREFIX
    line_json = json.loads(line)
    if line_json['response_code'] != 200 or not line_json['respbody'] or not line_json['raw_query']:
        return
    result_dict = dict()
    try:
        res_dict = parseResbody(body=line_json['respbody'])
        url = parseImage(raw_query=line_json['raw_query'])
        if res_dict is not None and url is not None:
            for key in res_dict:
                result_dict[key] = res_dict[key]
            result_dict['url'] = url
            result_dict['newImageName'] = IMAGE_PREFIX + \
                '{:0>8}'.format(str(IMAGE_COUNT)) + '.jpg'
            IMAGE_COUNT += 1
        RES_FILE.write(json.dumps(result_dict, ensure_ascii=False))
        RES_FILE.write('\n')
        RES_FILE.flush()
    except Exception as e:
        print(line_json)
        print(str(e))
        exit()


def downloadImageByResFile(file=None, imagePath=None):
    downloadImage_By_urllist(urlListFile=file, imageSaveDir=imagePath)
    pass


def parseFile(file=None):
    global RES_FILE
    global IMAGE_PREFIX
    global IMAGE_COUNT
    global CONFIG
    res_file = file.replace('origin', 'res')
    RES_FILE = open(res_file, 'w')
    IMAGE_PREFIX = CONFIG['prefix'].upper()+CONFIG['data']+'-'
    IMAGE_COUNT = 0
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            parseLine(line=line)
            # break
    RES_FILE.close()
    saveLogPath = '../saveLogDir'
    if not os.path.exists(saveLogPath):
        os.makedirs(saveLogPath)
    copyfile(res_file, os.path.join(saveLogPath, os.path.basename(res_file)))
    saveImagePath = '../saveImageDir'
    if not os.path.exists(saveImagePath):
        os.makedirs(saveImagePath)
    saveImagePath_date = os.path.join(saveImagePath,CONFIG['data'])
    if not os.path.exists(saveImagePath_date):
        os.makedirs(saveImagePath_date)
    downloadImageByResFile(file=res_file, imagePath=saveImagePath_date)


def initConfig():
    global CONFIG
    CONFIG = yaml.load(open('config.yaml'))
# def updataConfig(config=None):
#     conf = yaml.load(open('config.yaml'))
#     print(conf)
#     reu


def main():
    initConfig()
    print(CONFIG)
    logFile = downloadFile(CONFIG)
    parseFile(file=logFile)


if __name__ == '__main__':
    main()
