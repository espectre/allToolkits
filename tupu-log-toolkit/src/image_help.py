# -*- coding:utf-8 -*
import os
import sys
import json
import hashlib
import threading
import queue as Queue
import time
import numpy as np
import cv2


def downloadImage_By_urllist(urlListFile=None, imageSaveDir=None):
    wgetImageFromUrl_MulThread(
        urlFile=urlListFile, saveBasePath=imageSaveDir)
    # md5 process image file and rename file
    pass


GLOBAL_LOCK = threading.Lock()
ERROR_NUMBER = 0
IMAGE_SAVE_PATH = None
THREAD_DOWNLOAD_COUNT = 20


def wgetImageFromUrl_MulThread(urlFile=None, saveBasePath=None):
    # globale vars initialization
    global IMAGE_SAVE_PATH
    inputFileOp = open(urlFile, 'r')
    logErOp = open(urlFile+'-wget-error.log', 'w')
    IMAGE_SAVE_PATH = saveBasePath
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)
    queue = Queue.Queue(0)
    thread_prod = prod_worker(queue, inputFileOp)
    thread_prod.start()
    print('thread:', thread_prod.name, 'successfully started')
    time.sleep(10)
    for i in range(THREAD_DOWNLOAD_COUNT):
        exec('thread_cons_{} = cons_worker(queue,IMAGE_SAVE_PATH,logErOp)'.format(i))
        eval('thread_cons_{}.start()'.format(i))
    thread_prod.join()
    for i in range(THREAD_DOWNLOAD_COUNT):
        eval('thread_cons_{}.join()'.format(i))
    print('total error number:', ERROR_NUMBER)
    inputFileOp.close()
    logErOp.close()

    pass


class prod_worker(threading.Thread):
    """
    producing worker
    """
    global GLOBAL_LOCK

    def __init__(self, queue, infileOp):
        threading.Thread.__init__(self)
        self.queue = queue
        self.infileOp = infileOp

    def run(self):
        for line in self.infileOp:
            line = line.strip()
            if line == None or len(line) <= 0:
                continue
            GLOBAL_LOCK.acquire()
            self.queue.put(line)
            GLOBAL_LOCK.release()
        GLOBAL_LOCK.acquire()
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


class cons_worker(threading.Thread):
    global GLOBAL_LOCK
    global IMAGE_SAVE_PATH

    def __init__(self, queue, savePath, logErOp):
        threading.Thread.__init__(self)
        self.queue = queue
        self.savePath = savePath
        self.logErOp = logErOp

    def download(self, url, output_path):
        err_flag = 0
        url_json = json.loads(url)
        if 'label' not in url_json:
            return
        output_path = os.path.join(output_path,str(url_json['label']))
        if not os.path.exists(output_path):
            GLOBAL_LOCK.acquire()
            os.makedirs(output_path)
            GLOBAL_LOCK.release()
        image_save_path = os.path.join(output_path, url_json['newImageName'].split('.')[0])
        try:
            cmdStr = "wget %s -O %s -q" % (url_json['url'], image_save_path)
            ret = os.system(cmdStr)
            count = 5
            while ret != 0 and count > 0:
                ret = os.system(cmdStr)
                count -= 1
            if ret != 0:
                err_flag = 1
        except all as e:
            err_flag = 1
        if err_flag == 0:
            if os.path.exists(image_save_path) and (not  os.path.getsize(image_save_path)):
                os.remove(image_save_path)
                return 0
            cv2ImreadAndWrite(oldImageNamePath=image_save_path,
                              newImageNamePath=os.path.join(
                                  output_path, url_json['newImageName']))
            os.remove(image_save_path)
            if not os.path.getsize(os.path.join(output_path,url_json['newImageName'])):
                os.remove(os.path.join(output_path,url_json['newImageName']))
        #else:
        #    os.remove(image_save_path)

        return err_flag

    def run(self):
        global ERROR_NUMBER
        err_num = 0
        while(not self.queue.empty()):
            if GLOBAL_LOCK.acquire(False):
                # customized downloading code
                url = self.queue.get()
                GLOBAL_LOCK.release()
                err_flag = self.download(url, IMAGE_SAVE_PATH)
                if err_flag == 1:
                    # wget error
                    err_num += 1
                    GLOBAL_LOCK.acquire()
                    self.logErOp.write(url+'\n')
                    self.logErOp.flush()
                    GLOBAL_LOCK.release()
                    pass
            else:
                pass
        GLOBAL_LOCK.acquire()
        ERROR_NUMBER += err_num
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


def cv2ImreadAndWrite(oldImageNamePath=None, newImageNamePath=None):
    try:
        im = cv2.imread(oldImageNamePath, cv2.IMREAD_COLOR)
        cv2.imwrite(newImageNamePath, im)
        return True
    except:
        print("ERROR cv2 imwrite %s, %s" %
              (oldImageNamePath, newImageNamePath))
    return False
