inference_param_config={
    "modelParam":{ # model param of model
        "weight": "/workspace/data/BK/process-mp4/models/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512_final.caffemodel",
        "deploy": "/workspace/data/BK/process-mp4/models/ResNet/coco/refinedet_resnet101_512x512/deploy.prototxt",
        "label": "/workspace/data/BK/process-mp4/models/ResNet/coco/refinedet_resnet101_512x512/coco_label.csv",
        "batch_size":1
    },
    'imageSizeParam':{ # image size param
        'resize_size':None,
        'center_crop_size':None,
    },
    'imageDataParam':{ # input model data preProcess param
        'mean':None,
        'scala':None,
    },
    'need_label_class_thresholds_dict': { 
        # param of model output and postProcess param
        # key : index == model output class index
        1: ["guns",0.1],
        2: ["knives",0.1],
        3: ["tibetan flag",0.1],
        4: ["islamic flag",0.1],
        5: ["isis flag",0.1],
    },
}
