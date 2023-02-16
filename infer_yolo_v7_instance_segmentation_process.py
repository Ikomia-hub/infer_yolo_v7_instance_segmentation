# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
import copy
import os
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.general import non_max_suppression, \
    scale_coords
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.segment.general import scale_masks, \
    process_mask_upsample
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.dataloaders import letterbox
import torch
import numpy as np
import random
import yaml
import onnxruntime as ort
import cv2


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.weights = ""
        self.imgsz = 640
        self.thr_conf = 0.25
        self.iou_conf = 0.5
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.weights = str(param_map["weights"])
        self.imgsz = int(param_map["imgsz"])
        self.thr_conf = float(param_map["thr_conf"])
        self.iou_conf = float(param_map["iou_conf"])
        self.update = True

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["weights"] = str(self.weights)
        param_map["imgsz"] = str(self.imgsz)
        param_map["thr_conf"] = str(self.thr_conf)
        param_map["iou_conf"] = str(self.iou_conf)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add instance segmentation output
        self.addOutput(dataprocess.CInstanceSegIO())

        self.inst_seg_output = None
        self.weights = ""
        self.device = torch.device("cpu")
        self.session = None
        self.stride = 32
        self.thr_conf = 0.25
        self.iou_conf = 0.45
        self.classes = None
        self.colors = None
        self.providers = ['CPUExecutionProvider']
        self.coco_data =  os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "yolov7", "seg", 
                                                        "data", "coco_info.yaml")

        # Create parameters class
        if param is None:
            self.setParam(InferYoloV7InstanceSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, img0):
        # Get parameters :
        param = self.getParam()
        # Padded resize
        h, w = np.shape(img0)[:2]
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img, ratio, dwdh = letterbox(img, int(param.imgsz))

        # Convert
        img = img.transpose(2, 0, 1)  # HxWxC, to CxHxW
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        output_names = [x.name for x in self.session.get_outputs()]

        y = self.session.run(output_names, {self.session.get_inputs()[0].name: img})

        pred, *others, proto = [torch.tensor(i, device="cpu") for i in y] # return to torch
        y = (pred, (others, proto))

        nm = pred.shape[-1] - 5 - len(self.classes)

        pred = non_max_suppression(pred,
                                   param.thr_conf,
                                   param.iou_conf,
                                   None,
                                   False,
                                   max_det=100,
                                   nm=nm)

        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                masks = process_mask_upsample(proto[i],
                                              det[:, 6:],
                                              det[:, :4],
                                              img.shape[-2:])  # HWC
                masks = masks.permute(1, 2, 0).detach().cpu().numpy()
                masks = scale_masks(img.shape[-2:], masks, img0.shape, ratio_pad=None)
                masks = np.transpose(masks, (2, 0, 1))

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (h, w)).round()
                det = det.detach().cpu().numpy()

                for j, (mask, bbox_cls) in enumerate(zip(masks, det)):
                    cls = int(bbox_cls[5])
                    conf = float(bbox_cls[4])
                    x_obj, y_obj, w_obj, h_obj = bbox_cls[:4]
                    x_obj = float(x_obj)
                    y_obj = float(y_obj)
                    h_obj = float(h_obj) - y_obj
                    w_obj = float(w_obj) - x_obj
                    mask = mask.astype(dtype='uint8')
                    self.inst_seg_output.addInstance((i + 1) * j, 0, cls,
                                                     self.classes[cls],
                                                     conf, x_obj, y_obj,
                                                     w_obj, h_obj, mask,
                                                     self.colors[cls])

    def run(self):
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        # Get input :
        input = self.getInput(0)
        # Get image from input/output (numpy array):
        srcImage = input.getImage()
        # Get outputs :
        self.inst_seg_output = self.getOutput(1)
        h, w = np.shape(srcImage)[:2]
        self.inst_seg_output.init("YoloV7", 0, w, h)
        # Forward input image
        self.forwardInputImage(0, 0)
        # Get parameters :
        param = self.getParam()

        with open(self.coco_data) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            names = data["names"]

        if param.update or self.session is None:
            self.device = torch.device("cpu")
            print("Will run on {}".format(self.device.type))
            self.session = ort.InferenceSession(param.weights, providers=self.providers)
            self.classes = names
            random.seed(0)
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]
            # remove added path in pythonpath after loading model
            param.update = False

        # Call to the process main routine
        with torch.no_grad():
            self.infer(srcImage)

        self.setOutputColorMap(0, 1, [[0, 0, 0]] + self.colors)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v7_instance_segmentation"
        self.info.shortDescription = "Inference for YOLO v7 instance segmentation models"
        self.info.description = "Inference for YOLO v7 instance segmentation models in .onnx format"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Instance Segmentation"
        self.info.iconPath = "icons/yolov7.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
        self.info.journal = "arXiv preprint arXiv:2207.02696"
        self.info.year = 2022
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentationLink = "https://github.com/WongKinYiu/yolov7/tree/u7/seg"
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolov7/tree/u7/seg"
        # Keywords used for search
        self.info.keywords = "yolo, instance, segmentation, coco"

    def create(self, param=None):
        # Create process object
        return InferYoloV7InstanceSegmentation(self.info.name, param)
