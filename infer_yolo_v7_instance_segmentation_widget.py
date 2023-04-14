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
        self.spin_input_size = pyqtutils.append_spin(self.grid_layout, "Image size", self.parameters.input_size)
        self.spin_conf_thres = pyqtutils.append_double_spin(self.grid_layout, "Confidence threshold",
                                                          self.parameters.conf_thres,
                                                          min=0., max=1., step=0.01, decimals=2)
        self.spin_iou_thres = pyqtutils.append_double_spin(self.grid_layout, "Confidence IOU",
                                                          self.parameters.iou_thres,
                                                          min=0., max=1., step=0.01, decimals=2)
        self.check_use_custom_model = pyqtutils.append_check(self.grid_layout, "Custom train",
                                                         self.parameters.use_custom_model)
        self.check_use_custom_model.stateChanged.connect(self.on_use_custom_model_changed)
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model name")

        for name in model_zoo.keys():
            self.combo_model_name.addItem(name)

        self.combo_model_name.setCurrentText(self.parameters.model_name)
        self.combo_model_name.setEnabled(not self.parameters.use_custom_model)
        self.browse_model_path = pyqtutils.append_browse_file(self.grid_layout, "Custom model",
                                                                self.parameters.model_path)
        self.browse_model_path.setEnabled(self.parameters.use_custom_model)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_use_custom_model_changed(self, name):
        self.combo_model_name.setEnabled(not self.check_use_custom_model.isChecked())
        self.browse_model_path.setEnabled(self.check_use_custom_model.isChecked())

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.use_custom_model = self.check_use_custom_model.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.model_name = self.combo_model_name.currentText()
        self.parameters.iou_thres = self.spin_iou_thres.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.model_path = self.browse_model_path.path
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
