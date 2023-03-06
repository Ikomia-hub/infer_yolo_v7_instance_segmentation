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
from infer_yolo_v7_instance_segmentation.ikutils import download_model, clamp
import torch
from infer_yolo_v7_instance_segmentation.yolov7.seg.models.experimental import attempt_load
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.dataloaders import letterbox
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, clip_coords
from infer_yolo_v7_instance_segmentation.yolov7.seg.models.yolo import Model
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.torch_utils import torch_load
from infer_yolo_v7_instance_segmentation.yolov7.seg.utils.segment.general import process_mask, scale_masks, \
    process_mask_upsample
import numpy as np


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.img_size = 640
        self.custom_train = False
        self.pretrain_model = 'yolov7-seg'
        self.cuda = torch.cuda.is_available()
        self.thr_conf = 0.25
        self.iou_conf = 0.5
        self.custom_model = ""
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.img_size = int(param_map["img_size"])
        self.custom_train = utils.strtobool(param_map["custom_train"])
        self.pretrain_model = str(param_map["pretrain_model"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.thr_conf = float(param_map["thr_conf"])
        self.iou_conf = float(param_map["iou_conf"])
        self.custom_model = param_map["custom_model"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "custom_train": str(self.custom_train),
            "img_size": str(self.img_size),
            "pretrain_model": str(self.pretrain_model),
            "thr_conf": str(self.thr_conf),
            "iou_conf": str(self.iou_conf),
            "cuda": str(self.cuda),
            "custom_model": str(self.custom_model)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentation(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        self.model = None
        self.weights = ""
        self.device = torch.device("cpu")
        self.stride = 32
        self.imgsz = 640
        self.thr_conf = 0.25
        self.iou_conf = 0.45
        self.classes = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV7InstanceSegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, img0):
        h, w = np.shape(img0)[:2]
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose(2, 0, 1)  # HxWxC, to CxHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type == 'cuda' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred, out = self.model(img)
        proto = out[1]
        # number of masks
        nm = pred.shape[-1] - 5 - len(self.classes)
        pred = non_max_suppression(pred, self.thr_conf, self.iou_conf, None, False, max_det=100, nm=nm)

        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                masks = process_mask_upsample(proto[i], det[:, 6:], det[:, :4], img.shape[-2:])  # HWC
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
                    self.add_instance((i + 1) * j, 0, cls, conf, x_obj, y_obj, w_obj, h_obj, mask)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.begin_task_run()

        # Get input :
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # Get parameters :
        param = self.get_param_object()

        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.iou_conf = param.iou_conf
            self.thr_conf = param.thr_conf
            print("Will run on {}".format(self.device.type))

            if param.custom_train:
                ckpt = torch_load(param.custom_model, device=self.device)
                # custom model trained with ikomia
                if "yaml" in ckpt:
                    cfg = ckpt["yaml"]
                    self.classes = ckpt["names"]
                    state_dict = ckpt["state_dict"]
                    self.model = Model(cfg=cfg, ch=3, nc=len(self.classes), anchors=None)
                    self.model.load_state_dict(state_dict)
                    self.model.float().fuse().eval().to(self.device)
                    del ckpt
                # other
                else:
                    del ckpt
                    self.model = attempt_load(param.custom_model, device=self.device, fuse=True)  # load FP32 model
                    self.classes = self.model.names
            else:
                weights_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
                if not os.path.isdir(weights_folder):
                    os.mkdir(weights_folder)

                self.weights = os.path.join(weights_folder, param.pretrain_model + '.pt')
                if not os.path.isfile(self.weights):
                    download_model(param.pretrain_model, weights_folder)

                self.model = attempt_load(self.weights, device=self.device, fuse=True)  # load FP32 model
                self.classes = self.model.names

            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz = check_img_size(param.img_size, s=self.stride)  # check img_size

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))  # run once

            half = self.device.type != 'cpu'  # half precision only supported on CUDA
            if half:
                self.model.half()  # to FP16

            self.set_names(list(self.classes.values()))
            param.update = False

        # Call to the process main routine
        with torch.no_grad():
            self.infer(src_image)

        # Step progress bar:
        self.emit_step_progress()
        # Call endTaskRun to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v7_instance_segmentation"
        self.info.short_description = "Inference for YOLO v7 instance segmentation models"
        self.info.description = "Inference for YOLO v7 instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Instance Segmentation"
        self.info.icon_path = "icons/yolov7.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
        self.info.journal = "arXiv preprint arXiv:2207.02696"
        self.info.year = 2022
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://github.com/WongKinYiu/yolov7/tree/u7/seg"
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolov7/tree/u7/seg"
        # Keywords used for search
        self.info.keywords = "yolo, instance, segmentation, coco"

    def create(self, param=None):
        # Create process object
        return InferYoloV7InstanceSegmentation(self.info.name, param)
