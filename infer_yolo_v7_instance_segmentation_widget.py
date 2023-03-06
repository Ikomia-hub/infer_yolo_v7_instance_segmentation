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
from infer_yolo_v7_instance_segmentation.ikutils import model_zoo
from torch.cuda import is_available


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
        self.grid_layout = QGridLayout()

        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())
        self.spin_img_size = pyqtutils.append_spin(self.grid_layout, "Image size", self.parameters.img_size)
        self.spin_thr_conf = pyqtutils.append_double_spin(self.grid_layout, "Confidence threshold",
                                                          self.parameters.thr_conf,
                                                          min=0., max=1., step=0.01, decimals=2)
        self.spin_iou_conf = pyqtutils.append_double_spin(self.grid_layout, "Confidence IOU",
                                                          self.parameters.iou_conf,
                                                          min=0., max=1., step=0.01, decimals=2)
        self.check_custom_train = pyqtutils.append_check(self.grid_layout, "Custom train",
                                                         self.parameters.custom_train)
        self.check_custom_train.stateChanged.connect(self.on_custom_train_changed)
        self.combo_pretrain_model = pyqtutils.append_combo(self.grid_layout, "Model name")

        for name in model_zoo.keys():
            self.combo_pretrain_model.addItem(name)

        self.combo_pretrain_model.setCurrentText(self.parameters.pretrain_model)
        self.combo_pretrain_model.setEnabled(not self.parameters.custom_train)
        self.browse_custom_model = pyqtutils.append_browse_file(self.grid_layout, "Custom model",
                                                                self.parameters.custom_model)
        self.browse_custom_model.setEnabled(self.parameters.custom_train)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_custom_train_changed(self, name):
        self.combo_pretrain_model.setEnabled(not self.check_custom_train.isChecked())
        self.browse_custom_model.setEnabled(self.check_custom_train.isChecked())

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.custom_train = self.check_custom_train.isChecked()
        self.parameters.img_size = self.spin_img_size.value()
        self.parameters.pretrain_model = self.combo_pretrain_model.currentText()
        self.parameters.iou_conf = self.spin_iou_conf.value()
        self.parameters.thr_conf = self.spin_thr_conf.value()
        self.parameters.custom_model = self.browse_custom_model.path
        self.parameters.update = True
        # Send signal to launch the process
        self.emit_apply(self.parameters)


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
