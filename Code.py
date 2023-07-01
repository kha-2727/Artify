import cv2 
import numpy as np
import dlib

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def createBox(img,points):

    mask=np.zeros_like(img)
    mask=cv2.fillPoly(mask,[points],(255,255,255))
    img=cv2.bitwise_and(img,mask)
    
    bbox=cv2.boundingRect(points)
    x,y,w,h=bbox
    imgCrop=img[y:y+h,x:x+w]
    imgCrop=cv2.resize(imgCrop,(0,0),None,5,5)
    return mask
# Taking an image as input
img=cv2.imread('images.jpg')
# Resizing the image according to the requirement
img=cv2.resize(img,(0,0),None,0.5,0.5)
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
    imgOriginal=cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
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
    cv2.imshow('Lips',imgLips)
#     Coloring the lips
    imgColorLips=np.zeros_like(imgLips)
#     Lip color purpule
    imgColorLips[:]=153,0,157
    cv2.imshow('Colorlips',imgColorLips)
    
#     Taking and of color image with original image of lips
    imgColorLips=cv2.bitwise_and(imgLips,imgColorLips)
#     Taking Guassian blur to make it look more real
    imgColorLips=cv2.GaussianBlur(imgColorLips,(7,7),10)
#     Blending with original image so the lip color get changed on original image
    imgColorLips=cv2.addWeighted(imgOriginal,1,imgColorLips,0.4,0)
# Displaying colorLip image
    cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)

# Resize the window
    cv2.resizeWindow('Image Window', 600, 600)

# Display the image
    cv2.imshow('Image Window', img)
    cv2.imshow('ImageWindow',imgColorLips)


cv2.imshow('ImageWindow',imgOriginal)
cv2.waitKey(0)



img = cv2.imread('facefilter.jpg')
img=cv2.resize(img,(0,0),None,0.5,0.5)

# Load the 68 landmark detector model
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
    result = cv2.addWeighted(img, 1, cartoon_full, 0.9, 0)
    img = result
    
# Display the cartoonified image
cv2.imshow('Cartoonified Image', cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
