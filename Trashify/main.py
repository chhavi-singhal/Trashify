 
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import os
cap = cv2.VideoCapture(0)  # Initialize video capture
classifier = Classifier('Trashify/resources/Model/keras_model.h5', 'Trashify/resources/Model/labels.txt')

#import all waste images
# imgWasteList = []
# pathFolderWaste = "Trashify/resources/Waste"
# pathList = os.listdir(pathFolderWaste)
# for path in pathList:
#     imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

WasteList=[ "organic"
, "batteries"
, "paper"
, "plastic"
, "clothes"
,"bulb"
, "metal"
,"e-waste"
,"glass"
, ""]

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 2
   
# Blue color in BGR 
color = (0, 0, 0) 
  
# Line thickness of 2 px 
thickness = 5

while True:
    _, img = cap.read()  
    imgResize = cv2.resize(img,(454, 340))

    imgBackground= cv2.imread('Trashify/resources/background.png')

    predection = classifier.getPrediction(img)

    classID = predection[1]
    print(classID)
    if classID !=9 :
        textsize = cv2.getTextSize(WasteList[classID], font, 1, 2)[0] 
    
        textX = 2*(imgBackground.shape[1] - textsize[0]) // 3
        textY = (imgBackground.shape[0] + textsize[1]) // 2
        #imgBackground =  cvzone.overlayPNG(imgBackground, imgWasteList[classID-1],pos=[909, 127])
        imgBackground = cv2.putText(imgBackground, WasteList[classID], (textX, textY ) , font, fontScale, color, thickness, cv2.LINE_AA)
        predection= None
        #print(predection)

    
    #cv2.imwrite("img.jpg", output) 
    imgBackground[148:148+340, 159:159+454] = imgResize
    #displays
    #cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)  
