import numpy as np
from tensorflow.keras import models
from matplotlib import image
from cv2 import resize
from styles import dark_fusion, default
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QComboBox, QPlainTextEdit, QLineEdit
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QLabel, QVBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem

model = models.load_model('mstar.h5')
classes = {0:'BMP2', 1:'BTR70', 2:'T72'}

class app_home(QApplication):
	def __init__(self, arg):
		super().__init__(arg)

	def change_theme(self, dark):
		dark_fusion(self) if dark else default(self)

class main_window(QWidget):
	def __init__(self):
		super().__init__()
		self.setMinimumSize(640, 400)
		self.setWindowTitle("ASM_IDE: Making Assembly Language Easier")

	def change_size(self, size):
		if size == "Tiny":
			self.resize(640, 400)
		elif size == "Small":
			self.resize(800, 600)
		elif size == "Large":
			self.resize(1920, 1080)
		elif size == "Medium":
			self.resize(1280, 768)

class theme_select(QComboBox):
	def __init__(self, mode):
		super().__init__()
		if mode == "theme":
			self.addItems(["Dark", "Light"])
			self.currentIndexChanged.connect(self.themeChanged)
			self.setCurrentIndex(0)
			self.themeChanged()
		else:
			self.addItems(["Tiny", "Small", "Medium", "Large"])
			self.currentIndexChanged.connect(self.sizeChanged)
			self.setCurrentIndex(0)

	def themeChanged(self):
		if self.currentIndex() == 0:
			app.change_theme(True)
		else:
			app.change_theme(False)

	def sizeChanged(self):
		if self.currentIndex() == 0:
			window.change_size("Tiny")
		elif self.currentIndex() == 1:
			window.change_size("Small")
		elif self.currentIndex() == 2:
			window.change_size("Medium")
		else:
			window.change_size("Large")

class text_area(QPlainTextEdit):
	def __init__(self, read_only):
		super().__init__()
		self.setReadOnly(read_only)
		self.imageName = list()
		self.imageSet = list()
		#self.highlighter = SyntaxHighlighter(self.document())

	def commit(self, text, client):
		if client == "user":
			self.appendPlainText("You: " + text)
		elif client == "assistant":
			self.appendPlainText("Assistant: " + text)
		elif client == "system":
			self.appendPlainText(text)
		else:
			self.appendPlainText("Unauthorized Usage Detected")

	def upload(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;JPEG Files (*.jpeg);;JPG files (*.jpg);;PNG files (*.png)", options=options)
		if fileName:
			codefile = image.imread(fileName)
			get_image_name = list(fileName)
			temp_filename = list()
			while(1):
				char = get_image_name.pop()
				if char != '/':
					temp_filename.append(char)
				else:
					break
			self.imageName.append("".join(temp_filename[::-1]))
			self.imageSet.append(codefile)
		else:
			ai_comms.commit("Upload Failed, please try again", "assistant")
			ai_comms.commit("Uploaded Images: "+str(len(self.imageSet)), "system")
			return
		ai_comms.commit("Upload Successful", "assistant")
		ai_comms.commit("Uploaded Images: "+str(len(self.imageSet)), "system")

class Prediction_Output(QTableWidget):
	def __init__(self):
		super().__init__()
		self.numrows = 0
		self.setColumnCount(2)
		self.setHorizontalHeaderLabels(["ImageName", "Prediction"])
		row_index = 0

	def update(self, data):
		for row_data in data:
			self.insertRow(self.numrows)
			image_name = QTableWidgetItem(str(row_data[0]))
			pred = QTableWidgetItem(str(row_data[1]))
			self.setItem(self.numrows, 0, image_name)
			self.setItem(self.numrows, 1, pred)

	def reset(self):
		self.numrows = 0

def prediction():
	global model
	global classes
	print(type(model))
	test_set = np.array(ai_comms.imageSet)
	
	if len(ai_comms.imageSet) == 0:
		ai_comms.commit("No image uploaded. Please upload an image.", "assistant")
	elif len(ai_comms.imageSet) == 1:
		test_set = np.resize(test_set, [1,128,128,1])
	else:
		test_set = np.resize(test_set, [len(ai_comms.imageSet), 128, 128, 1])
	
	print(test_set.shape)
	get_predictions = list(model.predict_classes(test_set))
	get_class = lambda x: classes[x]
	pred_classes = [get_class(y) for y in get_predictions]

	data = list(zip(ai_comms.imageName, pred_classes))
	output_table.reset()
	output_table.update(data)

	ai_comms.imageName = list()
	ai_comms.imageSet = list()

app = app_home([])
window = main_window()

layout = QGridLayout()
ui_cust_layout = QHBoxLayout()
upload_layout = QHBoxLayout()
control_panel = QHBoxLayout()

theme_label = QLabel("Theme: ")
theme = theme_select("theme")
size_label = QLabel("Size: ")
size = theme_select("size")
ui_cust_layout.addWidget(theme_label)
ui_cust_layout.addWidget(theme)
ui_cust_layout.addWidget(size_label)
ui_cust_layout.addWidget(size)

upload_label = QLabel("Upload Image: ")
upload_image = QPushButton("Upload File")
upload_layout.addWidget(upload_label)
upload_layout.addWidget(upload_image)

predict_start = QPushButton("Predict")
clear_all = QPushButton("Clear")
control_panel.addWidget(predict_start)
control_panel.addWidget(clear_all)

ai_comms = text_area(True)
output_table = Prediction_Output()

layout.addLayout(ui_cust_layout, 0, 0)
layout.addLayout(upload_layout, 1, 0)
layout.addWidget(ai_comms, 2, 0)
layout.addLayout(control_panel, 3, 0)
layout.addWidget(output_table, 0, 1, 3, 1)

upload_image.clicked.connect(ai_comms.upload)
predict_start.clicked.connect(prediction)

window.setLayout(layout)
window.show()
app.exec_()