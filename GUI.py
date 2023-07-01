import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,uic, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QWidget,QStackedWidget, QLineEdit
from PyQt5.QtGui import QPixmap
import logos
import cv2
import numpy as np
import dlib
from PIL.ImageQt import ImageQt 
from PIL import Image
dbusername=[]
dbpass=[]
class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen,self).__init__()
        uic.loadUi("window.ui",self)
        self.login.clicked.connect(self.gotologin)
        self.create.clicked.connect(self.gotosignup)
        global dbusername
        global dbpass
        
    def gotologin(self):
        login = LoginFields()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def gotosignup(self):
        signup=SignupScreen()
        widget.addWidget(signup)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class LoginFields(QDialog):
    def __init__(self):
        super(LoginFields,self).__init__()
        uic.loadUi("loginfields.ui",self)
        self.login_2.clicked.connect(self.gotologin)
        self.passedit.setEchoMode(QLineEdit.Password)
        self.back.clicked.connect(self.gotomain)
        
    def gotologin(self):
        username = self.usernameedit.text()
        password = self.passedit.text()
        flag=0
        for i in range(len(dbusername)):
            if dbusername[i] == username and dbpass[i]==password:
                flag=1
                print(username)
                print(password)
                login = LoginScreen()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex()+1)
        if (flag==0):
            self.showerror.setText(f"Account doesn't exist")
    def gotomain(self):
        welcome = WelcomeScreen()
        widget.addWidget(welcome)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
        
def createBox(img,points):

    mask=np.zeros_like(img)
    mask=cv2.fillPoly(mask,[points],(255,255,255))
    img=cv2.bitwise_and(img,mask)
    
    bbox=cv2.boundingRect(points)
    x,y,w,h=bbox
    imgCrop=img[y:y+h,x:x+w]
    imgCrop=cv2.resize(imgCrop,(0,0),None,5,5)
    return mask
              
class LoginScreen(QDialog):
    def __init__(self):
        super(LoginScreen,self).__init__()
        loadUi("login.ui",self) 
        self.image.clicked.connect(self.selectimage)
        self.lipstick.clicked.connect(self.applylipstick)
        self.cartoon.clicked.connect(self.makcartoon)
        
        
        
    def selectimage(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp)')
        global img
        # Load the selected image and store it in an array
        if file_path:
            img = cv2.imread(file_path)
            # img=cv2.resize(img,(0,0),None,0.5,0.5)
            self.img_array = np.array(img)
            # Display the image in the label
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(file_path))
            self.image_label.setPixmap(pixmap)
            
    def makcartoon(self):
        # cartoone = MakeCartoon()
        # widget.addWidget(cartoone)
        # widget.setCurrentIndex(widget.currentIndex()+1)
        detector=dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the landmarks in the face
        rects = detector(gray,1)
        for (i, rect) in enumerate(rects):
            landmarks = predictor(gray, rect)
            
            # Extract the facial landmarks corresponding to the face region
            x, y, w, h = landmarks.part(0).x, landmarks.part(17).y, landmarks.part(16).x - landmarks.part(0).x, landmarks.part(8).y - landmarks.part(27).y
        
            # Create a mask for the face region using the extracted facial landmarks
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
            gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            blurImage = cv2.medianBlur(img[y:y+h, x:x+w], 1)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            # Apply the cartoonify effect to the face region
            cartoon = cv2.bilateralFilter(img[y:y+h, x:x+w], 9, 75, 75)
            cartoon = cv2.bitwise_and(cartoon, cartoon, mask = edges)
            # Create a new array with the same size as the input image and copy the cartoonified face region onto it
            cartoon_full = np.zeros_like(img)
            cartoon_full[y:y+h, x:x+w] = cartoon
            
            # Blend the cartoonified face region with the original image
            # result = cv2.addWeighted(img, 1, cartoon_full, 0.9, 0)
            # img = result
        cv2.namedWindow('ImageWindow', cv2.WINDOW_NORMAL)

# Resize the window
        cv2.resizeWindow('ImageWindow', 400, 400)

# Display the image
        cv2.imshow('ImageWindow', img)
        cv2.imshow('ImageWindow',cartoon)
        # Display the cartoonified image
        # cv2.imshow('Cartoonified Image', cartoon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    def applylipstick(self):
        # profile = ManageProfile()
        # widget.addWidget(profile)
        # widget.setCurrentIndex(widget.currentIndex()+1)
        detector=dlib.get_frontal_face_detector()
        predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # Taking an image as input
        # img=cv2.imread('images.jpg')
        # Resizing the image according to the requirement
        # img=cv2.resize(img,(0,0),None,0.5,0.5)
        # Creating a copy of original image
        imgOriginal=img.copy()
        # Converting image to grayscale
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # detecting face using frontal face detector
        faces=detector(imgGray)
        # array to get landmarks points
        myPoints=[]
        # Iterating through faces in an image
        for face in faces:
        #     Getting top bottom right left axis of the face
            x1,y1=face.left(),face.top()
            x2,y2=face.right(),face.bottom()
        #     Drawing bounding box around the face
            # imgOriginal=cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #     Getting landmarks from face
            landmarks=predictor(imgGray,face)
            for n in range (68):
        #         Getting all landmarks and stroing it in array
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                myPoints.append([x,y])
        #         Displaying landmarks value on the face 
        #         cv2.circle(imgOriginal,(x,y),5,(50,50,255),cv2.FILLED)
        #         cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)
            
            myPoints=np.array(myPoints)
        #         Creating box around lips
            imgLips=createBox(img,myPoints[48:60])
            # cv2.imshow('Lips',imgLips)
        #     Coloring the lips
            imgColorLips=np.zeros_like(imgLips)
        #     Lip color purpule
            imgColorLips[:]=153,0,157
            # cv2.imshow('Colorlips',imgColorLips)
            
        #     Taking and of color image with original image of lips
            imgColorLips=cv2.bitwise_and(imgLips,imgColorLips)
        #     Taking Guassian blur to make it look more real
            imgColorLips=cv2.GaussianBlur(imgColorLips,(7,7),10)
        #     Blending with original image so the lip color get changed on original image
            imgColorLips=cv2.addWeighted(imgOriginal,1,imgColorLips,0.4,0)
            # print(type(imgColorLips))
            cv2.namedWindow('ImageWindow', cv2.WINDOW_NORMAL)

# Resize the window
            cv2.resizeWindow('ImageWindow', 400, 400)

# Display the image
            cv2.imshow('ImageWindow', img)
            cv2.imshow('ImageWindow',imgColorLips)

        


        
class SignupScreen(QDialog):
    def __init__(self):
        super(SignupScreen,self).__init__()
        loadUi("signup.ui",self) 
        # self.passedit.setEchoMode(QtWidgets.QLineEdit.Password)  
        # self.confirmpassedit.setEchoMode(QtWidgets.QLineEdit.Password)  
        self.signupbut.clicked.connect(self.gotologinfields)
        self.signuppass.setEchoMode(QLineEdit.Password)
        self.signupcnfrmpass.setEchoMode(QLineEdit.Password)
        
        # self.back.clicked.connect(self.gotomain)
    def gotologinfields(self):
        signupusername=self.signupuser.text()
        signuppassword=self.signuppass.text()
        signupcnfrmpassword=self.signupcnfrmpass.text()
        if (signuppassword==signupcnfrmpassword):
            dbusername.append(signupusername)
            dbpass.append(signuppassword)
            loginf=LoginFields()
            widget.addWidget(loginf)
            widget.setCurrentIndex(widget.currentIndex()+1)  
        else:
            self.showerror.setText(f"Passwords not same")
    
    def manageprofile(self):
        user=self.usernameedit.text()
        password=self.passedit.text()
        confirmpass=self.confirmpassedit.text()
        
        
        if len(user)==0 or len(password)==0 or len(confirmpass)==0:
            self.error.setText("Please fill all fields")
            
        elif password!=confirmpass:
            self.error2.setText("Passwords donot match!!")
            
        else:
            manageprofile=ManageProfileScreen()
            widget.addWidget(manageprofile)
            widget.setCurrentIndex(widget.currentIndex()+1)
        
class ManageProfileScreen(QDialog):
    def __init__(self):
        super(ManageProfileScreen,self).__init__()
        loadUi("manageprofile.ui",self)
        self.accbutton.clicked.connect(self.gotoactions)
        self.profilepic.clicked.connect(self.gotoimageview)
        
    def gotoimageview(self):
        self.label=self.findChild(QLabel,"profileimage")
        
    def gotoactions(self):
        actions=ActionScreen()
        widget.addWidget(actions)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class FeedbackScreen(QDialog):
    def __init__(self):
        super(FeedbackScreen,self).__init__()
        loadUi("feedback.ui",self)
        self.back.clicked.connect(self.gotomain)
    def gotomain(self):
        actions=ActionScreen()
        widget.addWidget(actions)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class  ActionScreen(QDialog):
      def __init__(self):
        super(ActionScreen,self).__init__()
        loadUi("actions.ui",self)
        self.inputvideo.clicked.connect(self.gotoinputscreen)
        self.viewfeedback.clicked.connect(self.gotofeedbackpage)
      def gotoinputscreen(self):
          inputscreen=InputScreen()
          widget.addWidget(inputscreen)
          widget.setCurrentIndex(widget.currentIndex()+1)
      def gotofeedbackpage(self):
          feedback=FeedbackScreen()
          widget.addWidget(feedback)
          widget.setCurrentIndex(widget.currentIndex()+1) 
          

class InputScreen(QDialog):
    def __init__(self):
        super(InputScreen,self).__init__()
        loadUi("videoinput.ui",self)
        self.back.clicked.connect(self.gotomain)
    def gotomain(self):
        actions=ActionScreen()
        widget.addWidget(actions)
        widget.setCurrentIndex(widget.currentIndex()+1) 
    
           
          
        
app=QApplication(sys.argv)
widget=QStackedWidget()
welcome=WelcomeScreen()
widget.addWidget(welcome)  
widget.setFixedHeight(500)
widget.setFixedWidth(700)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")


    

    

           
    