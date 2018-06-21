import os
import sys
# set used caffe path
caffe_path = "/workspace/data/BK/terror-det-refineDet-Dir/refineDet_Dir/RefineDet/python"
sys.path.insert(0, caffe_path)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format
net = caffe_pb2.NetParameter()
deploy_file = "/workspace/data/BK/terror-online/0620/merge_bn_model/test_change/deploy_change.prototxt"
with open(deploy_file) as f:
    s = f.read()
    text_format.Merge(s, net)
layerNames = [l.name for l in net.layer]
for layer_name in layerNames:
    print(layer_name)
idx = layerNames.index('data')
if idx != 0:
    print("error")
else:
    print("success")
data_layer = net.layer[idx]
#print(data_layer.input_param.shape[0])
data_layer.input_param.shape[0].dim[0] = 1
#print(data_layer.input_param.shape[0].dim[0])
#deploy_file_new = "/workspace/data/BK/terror-online/0620/merge_bn_model/test_change/deploy_change_1.prototxt"

with open(deploy_file, 'w') as f:
    f.write(str(net))
# l = net.layer[idx]
# l.param[0].lr_mult = 1.3

# outFn = '/tmp/newNet.prototxt'
# print 'writing', outFn
# with open(outFn, 'w') as f:
# f.write(str(net))
