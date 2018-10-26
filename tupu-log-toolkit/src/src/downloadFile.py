#coding=utf-8
import os
import sys


def getFileName(bucketListFile=None,config=None):
    successFlag = False
    resultFileNam = None
    with open(bucketListFile,'r') as f:
        for line in f:
            line_file = line.strip().split()[0]
            if config['data'] in line_file and "_SUCCESS" in line_file:
                successFlag = True
                continue
            if successFlag and config['data'] in line_file and config['prefix'] in line_file:
                resultFileNam = line_file
    return successFlag,resultFileNam

def downloadFile(config=None):
    qrsctlLoginDoralog = config['qrsctlDoralogLoginPath']
    res = os.system(qrsctlLoginDoralog)
    if res != 0:
        print("%s error" % (qrsctlLoginDoralog))
    qshellLoginDoralog = config['qshellDoralogLoginPath']
    res = os.system(qshellLoginDoralog)
    if res != 0:
        print("%s error" % (qshellLoginDoralog))
    fileBasePath = '../log-cache'
    if not os.path.exists(fileBasePath):
        os.makedirs(fileBasePath)
    bucketListFile = os.path.join(fileBasePath,'fileList.list')
    getBucketFileListCmd = "%s listbucket %s %s" % (
        config['qshellPath'], config['bucket'], bucketListFile)
    res = os.system(getBucketFileListCmd)
    if res != 0:
        print("%s error" % (getBucketFileListCmd))
    res, fileName = getFileName(bucketListFile=bucketListFile,config=config)
    if not res:
        print("%s file not success"%(config['data']))
        exit()
    logBasePath = '../logFileDir'
    if not os.path.exists(logBasePath):
        os.makedirs(logBasePath)
    logFilePath = os.path.join(logBasePath,config['data'])
    if not os.path.exists(logFilePath):
        os.makedirs(logFilePath)
    downloadlogFile = os.path.join(logFilePath,config['data']+'-origin.json')
    if os.path.exists(downloadlogFile):
        os.remove(downloadlogFile)
    downFileCmd = "%s get %s %s %s" % (
        config['qrsctlPath'], config['bucket'],fileName, downloadlogFile)
    res = os.system(downFileCmd)
    if res != 0:
        print("%s error" % (downFileCmd))
    return downloadlogFile
    pass

