import cv2 as cv
import mediapipe as mp
import time

from mediapipe.python.solutions.face_detection import FaceDetection 

cap = cv.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    
    results = faceDetection.process(img)
    #print(results)
    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection)
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.xmin * ih) , \
            int(bboxC.xmin * iw), int(bboxC.xmin * ih)
            cv.rectangle(img, bbox, (1000,0,1000),1) 
            cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0], bbox[1] - 20),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),2)      

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime 
    cv.putText(img,f'FPS:{int(fps)}', (5,25), cv.FONT_HERSHEY_COMPLEX,1,(50,0,1000),2)


    #blob = cv.dnn.blobFromImage(img,1/255,(wXh,wXh),[0,0,0],1,crop=False)
    #net.setInput(blob)

    #layerNames=net.getLayerNames()
    #outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    
    #outputs = net.forward(outputNames)
    cv.imshow('Window',img) 
    cv.waitKey(1)
