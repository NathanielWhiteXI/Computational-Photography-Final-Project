import sys
import backend
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QFileDialog, QToolBar, QLineEdit, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QAction, QImage
from PyQt6.QtCore import Qt


class ImageApp(QMainWindow):
    
    #Holds prior images (to allow for non destructive workflow).
    imageHistory = []

    #Value used to access current image.
    imageCounter = 0

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simple Image Restorer")

        #Central widget and layout
        c_widget = QWidget()
        layout = QVBoxLayout(c_widget)

        #Label to display the image
        self.label = QLabel("No image loaded")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        #User toggling section
        ui_widget = QWidget()
        ui_layout = QHBoxLayout(ui_widget)

        self.intensity_label = QLabel("Intensity (whole numbers only)")
        ui_layout.addWidget(self.intensity_label)

        self.intensity_box = QLineEdit("5")
        ui_layout.addWidget(self.intensity_box)

        layout.addWidget(ui_widget)

        self.setCentralWidget(c_widget)

        #Store original pixmap so we can rescale it
        self.original_pixmap = None

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        #Create actions (buttons)

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_image)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_image)

        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.clear_image)

        median_blur_action = QAction("Median Blur", self)
        median_blur_action.triggered.connect(self.median_blur_image)

        gaussian_blur_action = QAction("Gaussian Blur", self)
        gaussian_blur_action.triggered.connect(self.gaussian_blur_image)

        hist_equal_action = QAction("Histogram Equalization", self)
        hist_equal_action.triggered.connect(self.hist_eq_image)

        unsharp_mask_action = QAction("Unsharp Mask", self)
        unsharp_mask_action.triggered.connect(self.unsharpenMask_image)

        pipeline_action = QAction("Pipeline", self)
        pipeline_action.triggered.connect(self.image_pipeline)

        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.undo_image)

        #Add actions to toolbar
        toolbar.addAction(open_action)
        toolbar.addAction(save_action)
        toolbar.addAction(clear_action)
        toolbar.addAction(undo_action)
        toolbar.addAction(pipeline_action)
        toolbar.addAction(median_blur_action)
        toolbar.addAction(gaussian_blur_action)
        toolbar.addAction(hist_equal_action)
        toolbar.addAction(unsharp_mask_action)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.original_pixmap = QPixmap(file_name)
            
            #Add to workflow.
            self.imageHistory.append(self.qt_to_Numpy())
            self.imageCounter += 1

            self.update_image()

    def save_image(self):
        if self.original_pixmap:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            if file_name:
                self.original_pixmap.save(file_name)

    def undo_image(self):
        if self.imageCounter <= 1:
            return
        
        #Set the image to the prior image array.
        undidArray = self.imageHistory[self.imageCounter-2]
        undidPixMap = self.numpy_To_Qt(undidArray)
        self.original_pixmap = undidPixMap

        #Update workflow
        self.imageHistory.pop()
        self.imageCounter -= 1

        self.update_image()

        print("Image Undone") 

    def median_blur_image(self):
        if self.original_pixmap == None:
            return
            
        #Create numpy array out of image.
        npArray = self.qt_to_Numpy()

        #Apply effect to numpy array
        median_blur_npArray = backend.imgDenoising_Median(npArray, self.getInput())
        
        #Convert it back to normal.
        median_blur_pixmap = self.numpy_To_Qt(median_blur_npArray)
        self.original_pixmap = median_blur_pixmap

        #Add to workflow.
        self.imageHistory.append(self.qt_to_Numpy())
        self.imageCounter += 1

        self.update_image()

        print("Median Blur Complete")

    def gaussian_blur_image(self):
        if self.original_pixmap == None:
            return
            
        #Create numpy array out of image.
        npArray = self.qt_to_Numpy()

        #Apply effect to numpy array
        gaussian_blur_npArray = backend.imgDenoising_Gaussian(npArray, self.getInput())
        
        #Convert it back to normal.
        gaussian_blur_npArray = self.numpy_To_Qt(gaussian_blur_npArray)
        self.original_pixmap = gaussian_blur_npArray

        #Add to workflow.
        self.imageHistory.append(self.qt_to_Numpy())
        self.imageCounter += 1

        self.update_image()

        print("Gaussian Blur Complete")

    def hist_eq_image(self):
        if self.original_pixmap == None:
            return
            
        #Create numpy array out of image.
        npArray = self.qt_to_Numpy()

        #Apply effect to numpy array
        hist_eq_npArray = backend.histEq(npArray)
        
        #Convert it back to normal.
        hist_eq_npArray = self.numpy_To_Qt(hist_eq_npArray)
        self.original_pixmap = hist_eq_npArray

        #Add to workflow.
        self.imageHistory.append(self.qt_to_Numpy())
        self.imageCounter += 1

        self.update_image()

        print("Histogram Equalization Complete")

    def unsharpenMask_image(self):
        if self.original_pixmap == None:
            return
            
        #Create numpy array out of image.
        npArray = self.qt_to_Numpy()

        #Apply effect to numpy array
        unsharp_Mask_npArray = backend.unsharpMask(npArray, self.getInput())
        
        #Convert it back to normal.
        unsharp_Mask_npArray = self.numpy_To_Qt(unsharp_Mask_npArray)
        self.original_pixmap = unsharp_Mask_npArray

        #Add to workflow.
        self.imageHistory.append(self.qt_to_Numpy())
        self.imageCounter += 1

        self.update_image()

        print("Unsharpen Mask Complete")

    def image_pipeline(self):
        if self.original_pixmap == None:
            return
            
        #Create numpy array out of image.
        npArray = self.qt_to_Numpy()

        #Apply effect to numpy array
        pipeline_npArray = backend.run_auto_pipeline(npArray, self.getInput())
        
        #Convert it back to normal.
        pipeline_npArray = self.numpy_To_Qt(pipeline_npArray)
        self.original_pixmap = pipeline_npArray

        #Add to workflow.
        self.imageHistory.append(self.qt_to_Numpy())
        self.imageCounter += 1

        self.update_image()

        print("Pipeline Complete")

    def qt_to_Numpy(self):
        #Convert to image
        qImage = self.original_pixmap.toImage()

        #Converts to RGBA Image (what the backend uses)
        if qImage.format() != QImage.Format.Format_RGBA8888:
            qImage = qImage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = qImage.width()
        height = qImage.height()
        bytes_per_line = qImage.bytesPerLine()

        #Get the raw data to convert directly.
        ptr = qImage.bits()
        ptr.setsize(height * bytes_per_line)
        raw_data = ptr.asstring()

        numpy_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
        return numpy_array
    
    def numpy_To_Qt(self, array):
        height, width, channels = array.shape
        bytesPerLine = width * channels # 4 bytes
        qImage = QImage(array.data, width, height, bytesPerLine, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qImage)

    def clear_image(self):
        self.original_pixmap = None
        self.label.setText("No image loaded")

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def update_image(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.label.setPixmap(scaled)

            print(self.imageCounter)

    def getInput(self):
        return int(self.intensity_box.text())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec())
