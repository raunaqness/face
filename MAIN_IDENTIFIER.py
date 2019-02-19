#Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import cv2
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
import os, time, json

# Global Variables
identify = True
record = False
record_count = 0

# Variables for Face detection
id=0
font=cv2.FONT_HERSHEY_COMPLEX

fn_haar = 'haarcascade_frontalface_alt.xml'
fn_dir = 'att_faces'
im_width= 224
im_height=224
faceDetect =cv2.CascadeClassifier(fn_haar)

# Custom Modules
def find_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = faceDetect.detectMultiScale(gray,1.3,5)

    faces = faceDetect.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        legend_json = open("legend.json", "r")
        legend = json.load(legend_json)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        rec=cv2.face.FisherFaceRecognizer_create()
        read_test=rec.read('test.yml')
        id, conf = rec.predict(face_resize)

        person = legend[str(id)]
        if(person is None):
            person = str(id)

        print("Person : ", person)
        cv2.putText(frame,str(person),(x,y+h),font,1,(50,205,50),2,cv2.LINE_4)

    return(frame)


class ShowVideo(QtCore.QObject):
 
    #initiating the built in camera
    camera_port = 0
    # camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    
 
    def __init__(self, parent = None):

        self.KAFKA_BROKERS = '127.0.0.1:9092'
        self.KAFKA_TOPIC = 'test'

        # self.consumer = KafkaConsumer(self.KAFKA_TOPIC, bootstrap_servers=self.KAFKA_BROKERS, api_version=(2, 11, 2))
        super(ShowVideo, self).__init__(parent)
 
    @QtCore.pyqtSlot()
    def startVideo(self):

        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()


        # for msg in self.consumer:
        #     image = cv2.imdecode(np.frombuffer(msg.value, np.uint8), 1)
            # print("image_from_webcam : ", image.shape)

            global identify
            global record

            # print("identify : ", identify, " | record : ", record)
            # FACE IDENTIFICATION
            if(identify is True and record is False):
                frame_with_face = find_face(image)
                if frame_with_face is not None:
                    # print("Face Found")
                    image = frame_with_face

            # RECORD NEW FACE
            elif(record is True and identify is False):
                pass

            # Some Issue
            else:
                print("GLOBAL VARIABLES NOT SET PROPERLY")

            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, _ = color_swapped_image.shape
            
            #width = camera.set(CAP_PROP_FRAME_WIDTH, 1600)
            #height = camera.set(CAP_PROP_FRAME_HEIGHT, 1080)
            #camera.set(CAP_PROP_FPS, 15)

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)



class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
 
 
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QtGui.QImage()
 
    def initUI(self):
        self.setWindowTitle('Test')
 
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")
 
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()
 

vid = ShowVideo()




# Helper Functions
def record_new():
    global record
    global identify

    global vid

    identify = not identify
    record = not record

    # print("identify : ", identify)
    # print("record : ", record)

    if(identify is False):
        vid.record_button.setText("Stop Recording")
    else:
        vid.record_button.setText("Start Recording")

if __name__ == '__main__':
 
    app = QtWidgets.QApplication(sys.argv)
    thread = QtCore.QThread()
    thread.start()
    
    vid.moveToThread(thread)
    image_viewer = ImageViewer()
 
    vid.VideoSignal.connect(image_viewer.setImage)
 
    #Button to start the videocapture:
 
    push_button1 =QtWidgets.QPushButton('Start')
    push_button2 = QtWidgets.QPushButton('Test')
    push_button1.clicked.connect(vid.startVideo)
    vertical_layout = QtWidgets.QVBoxLayout()

    record_button = QtWidgets.QPushButton('Record')
    record_button.clicked.connect(record_new)

    vertical_layout.addWidget(image_viewer)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(record_button)
    # vertical_layout.addWidget(push_button2)
 
    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)
 
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())