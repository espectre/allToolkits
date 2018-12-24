# -*- coding:utf-8 -*-
import json
detect_labelx_json_dict = '{"url": "", "type": "image", "label": [{"name": "general", "type": "detection", "version": "1", "data": []}]}'

def format_url(url=None):
    d = eval(detect_labelx_json_dict)
    if len(url) > 256:
        print(url)
    d['url'] = url
    return d

def main():
    urlPrefix = "http://phe05pht1.bkt.clouddn.com/images/2Dbarcode/1030/"
    # urlPrefix = "http://phe05pht1.bkt.clouddn.com/images/barcode/1030/"
    urlFileList = "/Users/wangbing/QiNiuWordDir/labelx-projects/barcode-Dir/2Dbarcode.list"
    with open(urlFileList,'r') as f,open(urlFileList+'-labelx.json','w') as w_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            url = urlPrefix + line
            res = format_url(url=url)
            w_f.write(json.dumps(res)+'\n')
        pass
    

if __name__ == '__main__':
    main()
