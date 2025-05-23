import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from .nets.retinaface import RetinaFace
from .utils.anchors import Anchors
from .utils.config import cfg_mnet, cfg_re50
from .utils.utils import letterbox_image, preprocess_input
from .utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)


#------------------------------------#
#   Note the correspondence between the backbone network and pretrained weights
#   Be sure to modify model_path and backbone accordingly
#------------------------------------#
class Retinaface(object):
    _defaults = {
        #---------------------------------------------------------------------#
        #   To use your own trained model for prediction, you must modify model_path
        #   model_path points to the weight file in the logs folder
        #   After training, multiple weight files will be in the logs folder; choose the one with the lowest loss
        #---------------------------------------------------------------------#
        "model_path"        : 'model_data/Retinaface_resnet50.pth',
        #---------------------------------------------------------------------#
        #   The backbone network used: mobilenet, resnet50
        #---------------------------------------------------------------------#
        "backbone"          : 'resnet',
        #---------------------------------------------------------------------#
        #   Only bounding boxes with a score greater than the confidence threshold will be retained
        #---------------------------------------------------------------------#
        "confidence"        : 0.7,
        #---------------------------------------------------------------------#
        #   The IoU threshold used for non-maximum suppression
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,
        #---------------------------------------------------------------------#
        #   Whether to limit the image size
        #   When enabled, the input image size will be restricted to input_shape. Otherwise, the original image is used for prediction
        #   input_shape can be adjusted based on the input image size, must be a multiple of 32, e.g., [640, 640, 3]
        #---------------------------------------------------------------------#
        "input_shape"       : [1280, 1280, 3],
        #---------------------------------------------------------------------#
        #   Whether to limit the image size
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #--------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU is available
        #--------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize Retinaface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   Configuration information for different backbone networks
        #---------------------------------------------------#
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        #---------------------------------------------------#
        #   Generate anchor boxes
        #---------------------------------------------------#
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   Load model and weights
        #-------------------------------#
        self.net    = RetinaFace(cfg=self.cfg, mode='eval').eval()

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   Create a backup of the input image for later use in drawing
        #---------------------------------------------------#
        old_image = image.copy()
        #---------------------------------------------------#
        #   Convert the image to numpy format
        #---------------------------------------------------#
        image = np.array(image, np.float32)
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   Calculate scale to convert predicted boxes to the original image dimensions
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image adds gray bars to achieve distortion-free resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   Preprocess the image, normalize
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   Pass through the network for prediction
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            
            #-----------------------------------------------------------#
            #   Decode the predicted boxes
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   Obtain the confidence scores of the predictions
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   Decode facial landmarks
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   Stack the face detection results
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return old_image

            #---------------------------------------------------------#
            #   If letterbox_image is used, remove the gray bars
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        faces = []
        for b in boxes_conf_landms:
            x1, y1, x2, y2, confidence = b[:5].tolist()
            face = {
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence)
            }
            faces.append(face)
        
        return faces

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   Convert the image to numpy format
        #---------------------------------------------------#
        image = np.array(image, np.float32)
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)

        #---------------------------------------------------------#
        #   letterbox_image adds gray bars to achieve distortion-free resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            #-----------------------------------------------------------#
            #   Preprocess the image, normalize
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   Pass through the network for prediction
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            #-----------------------------------------------------------#
            #   Decode the predicted boxes
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   Obtain the confidence scores of the predictions
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   Decode facial landmarks
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   Stack the face detection results
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   Pass through the network for prediction
                #---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                #-----------------------------------------------------------#
                #   Decode the predicted boxes
                #-----------------------------------------------------------#
                boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                #-----------------------------------------------------------#
                #   Obtain the confidence scores of the predictions
                #-----------------------------------------------------------#
                conf    = conf.data.squeeze(0)[:, 1:2]
                #-----------------------------------------------------------#
                #   Decode facial landmarks
                #-----------------------------------------------------------#
                landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                #-----------------------------------------------------------#
                #   Stack the face detection results
                #-----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def get_map_txt(self, image):
        #---------------------------------------------------#
        #   Convert the image to numpy format
        #---------------------------------------------------#
        image = np.array(image, np.float32)
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   Calculate scale to convert predicted boxes to the original image dimensions
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image adds gray bars to achieve distortion-free resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   Preprocess the image, normalize
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   Pass through the network for prediction
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            #-----------------------------------------------------------#
            #   Decode the predicted boxes
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   Obtain the confidence scores of the predictions
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   Decode facial landmarks
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   Stack the face detection results
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return np.array([])

            #---------------------------------------------------------#
            #   If letterbox_image is used, remove the gray bars
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        return boxes_conf_landms