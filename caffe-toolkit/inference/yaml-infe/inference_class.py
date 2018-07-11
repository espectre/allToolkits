import multiprocessing
import urllib
import json
import time


class Producer_Of_ImageNameQueue(multiprocessing.Process):
    def __init__(self, imageNameQueue, paramDictJsonStr, threadName):
        multiprocessing.Process.__init__(self)
        self.imageNameQueue = imageNameQueue
        self.paramDict = json.loads(paramDictJsonStr)
        self.threadName = threadName

    def getTimeFlag(self):
        return time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())

    def run(self):
        print("LOGINFO---%s---Thread %s begin running" %
              (self.getTimeFlag(), self.threadName))
        fileName = self.paramDict['inputFileName']
        beginIndex = int(self.paramDict['beginIndex'])
        with open(fileName, 'r') as f:
            for line in f.readlines()[beginIndex:]:
                line = line.strip()
                if len(line) <= 0:
                    continue
                self.imageNameQueue.put(line)
        for i in range(int(self.paramDict['imageDataProducerCount'])):
            self.imageNameQueue.put(None)
        print("LOGINFO---%s---Thread %s end" %
              (self.getTimeFlag(), self.threadName))
        pass


class Producer_Of_ImageDataQueue_And_consumer_Of_imageNameQueue(multiprocessing.Process):
    def __init__(self, imageNameQueue, imageDataQueue, paramDictJsonStr, threadName):
        multiprocessing.Process.__init__(self)
        self.imageNameQueue = imageNameQueue
        self.imageDataQueue = imageDataQueue
        self.paramDict = json.loads(paramDictJsonStr)
        self.urlFlag = True
        self.threadName = threadName

    def getTimeFlag(self):
        return time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())

    def readImage_fun(self, isUrlFlag=None, imagePath=None):
        """
            isUrlFlag == True , then read image from url
            isUrlFlag == False , then read image from local path
        """
        im = None
        if isUrlFlag == True:
            try:
                data = urllib.urlopen(imagePath.strip()).read()
                nparr = np.fromstring(data, np.uint8)
                if nparr.shape[0] < 1:
                    im = None
            except:
                im = None
            else:
                try:
                    im = cv2.imdecode(nparr, 1)
                except:
                    im = None
            finally:
                return im
        else:
            im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if np.shape(im) == ():
            return None
        return im

    def run(self):
        print("LOGINFO---%s---Thread %s begin running" %
              (self.getTimeFlag(), self.threadName))
        # self.urlFlag=self.paramDict['urlFlag']
        timeout_count = 0
        while True:
            try:
                imagePath = self.imageNameQueue.get(block=True, timeout=120)
            except:
                print("%s : %s  get timeout" %
                      (self.getTimeFlag(), self.threadName))
                timeout_count += 1
                if timeout_count > 5:
                    print("LOGINFO---%s---Thread exception,so kill %s" %
                          (self.getTimeFlag(), self.threadName))
                    break
                continue
            else:
                if imagePath == None:
                    print("LOGINFO---%s---Thread %s Exiting" %
                          (self.getTimeFlag(), self.threadName))
                    break
                imgData = self.readImage_fun(
                    isUrlFlag=self.urlFlag, imagePath=imagePath)
                if np.shape(imgData) == () or len(np.shape(imgData)) != 3 or np.shape(imgData)[-1] != 3:
                    print("WARNING---%s---imagePath %s can't read" %
                          (self.getTimeFlag(), imagePath))
                else:
                    self.imageDataQueue.put([imagePath, imgData])
                    pass
        self.imageDataQueue.put(None)
        print("LOGINFO---%s---Thread %s end" %
              (self.getTimeFlag(), self.threadName))
    pass


class Consumer_Of_ImageDataQueue_Inference(multiprocessing.Process):
    def __init__(self, imageDataQueue, paramDictJsonStr, threadName):
        multiprocessing.Process.__init__(self)
        self.imageDataQueue = imageDataQueue
        self.paramDict = json.loads(paramDictJsonStr)
        self.threadName = threadName
        self.saveFileOp = None
        self.gpuId = None
        self.modelFileName = None
        self.deployFileName = None
        self.labelFileName = None
        self.net = None
        self.label_list = None

    def getTimeFlag(self):
        return time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())

    def preInitial(self):
        self.saveFileOp = open(self.paramDict['saveResultFileName'], 'w')
        self.gpuId = int(self.paramDict['gpuId'])
        self.modelFileName = self.paramDict['modelFileName']
        self.deployFileName = self.paramDict['deployFileName']
        self.labelFileName = self.paramDict['labelFileName']
        self.image_size = self.paramDict['imagSize']

    def initalNetModel(self):
        caffe.set_mode_gpu()
        caffe.set_device(self.gpuId)
        self.net = caffe.Net(str(self.deployFileName),
                             str(self.modelFileName), caffe.TEST)
        with open(str(self.labelFileName), 'r') as f:
            self.label_list = caffe_pb2.LabelMap()
            text_format.Merge(str(f.read()), self.label_list)

    def preProcess(self, oriImage=None):
        img = cv2.resize(oriImage, (self.image_size, self.image_size))
        img = img.astype(np.float32, copy=False)
        img = img - np.array([[[103.52, 116.28, 123.675]]])
        img = img * 0.017
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        return img

    def postProcess(self, output=None, imagePath=None, height=None, width=None):
        """
            postprocess net inference result
        """
        w = width
        h = height
        bbox = output[0, :, 3:7] * np.array([w, h, w, h])
        cls = output[0, :, 1]
        conf = output[0, :, 2]
        result_dict = dict()
        result_dict['bbox'] = bbox.tolist()
        result_dict['cls'] = cls.tolist()
        result_dict['conf'] = conf.tolist()
        result_dict['imagePath'] = imagePath
        self.saveFileOp.write(json.dumps(result_dict)+'\n')
        self.saveFileOp.flush()

    def inference_fun(self, orginalImgData=None, imagePath=None):
        imgDataHeight = orginalImgData.shape[0]
        imgDataWidth = orginalImgData.shape[1]
        imgData = self.preProcess(orginalImgData)
        self.net.blobs['data'].data[...] = imgData
        output = self.net.forward()
        self.postProcess(output=output['detection_out'][0], imagePath=imagePath,
                         height=imgDataHeight, width=imgDataWidth)

    def run(self):
        print("LOGINFO---%s---Thread %s begin running" %
              (self.getTimeFlag(), self.threadName))
        self.preInitial()
        self.initalNetModel()
        endGetImageDataThreadCount = 0
        time_out_count = 0
        while True:
            # print("debug : %s   %s" % (str(self.imageDataQueue.qsize()),
            #                            str(self.imageDataQueue.empty())))
            try:
                next_imageData = self.imageDataQueue.get(
                    block=True, timeout=120)
            except:
                print("%s  get timeout" % (self.threadName))
                time_out_count += 1
                if endGetImageDataThreadCount >= self.paramDict['imageDataProducerCount'] or time_out_count > 8:
                    print("LOGINFO---%s---Thread Exception so kill  %s " %
                          (self.getTimeFlag(), self.threadName))
                    break
                continue
            else:
                if next_imageData == None:
                    endGetImageDataThreadCount += 1
                    if endGetImageDataThreadCount >= self.paramDict['imageDataProducerCount']:
                        print("LOGINFO---%s---Thread %s Exiting" %
                              (self.getTimeFlag(), self.threadName))
                        break
                else:
                    imagePath = next_imageData[0]
                    orginalImgData = next_imageData[1]
                    self.inference_fun(
                        orginalImgData=orginalImgData, imagePath=imagePath)
        print("LOGINFO---%s---Thread %s end" %
              (self.getTimeFlag(), self.threadName))
    pass
