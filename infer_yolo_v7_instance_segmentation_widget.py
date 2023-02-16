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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_yolo_v7_instance_segmentation.infer_yolo_v7_instance_segmentation_process import InferYoloV7InstanceSegmentationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYoloV7InstanceSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Image size
        self.spin_imgsz = pyqtutils.append_double_spin(self.gridLayout, "Input size",
                                                          self.parameters.imgsz,
                                                          min=0., max=2000, step=32, decimals=2)

        # Confidence threshold
        self.spin_thr_conf = pyqtutils.append_double_spin(self.gridLayout, "Confidence threshold",
                                                          self.parameters.thr_conf,
                                                          min=0., max=1., step=0.01, decimals=2)
        # Confidence IOU
        self.spin_iou_conf = pyqtutils.append_double_spin(self.gridLayout, "Confidence IOU",
                                                          self.parameters.iou_conf,
                                                          min=0., max=1., step=0.01, decimals=2)
        # Model path (.onnx)
        self.browse_model = pyqtutils.append_browse_file(self.gridLayout,
                                                label="Model path (.onnx)",
                                                path=self.parameters.weights,
                                                mode=QFileDialog.ExistingFile)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Get parameters from widget
        self.parameters.weights = self.browse_model.path
        self.parameters.imgsz = self.spin_imgsz.value()
        self.parameters.iou_conf = self.spin_iou_conf.value()
        self.parameters.thr_conf = self.spin_thr_conf.value()
        self.parameters.update = True
        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferYoloV7InstanceSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolo_v7_instance_segmentation"

    def create(self, param):
        # Create widget object
        return InferYoloV7InstanceSegmentationWidget(param, None)
