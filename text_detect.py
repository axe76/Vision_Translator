import numpy as np
import cv2

def detect(path):

    image = cv2.imread(path)
    orig = image.copy()
    (H, W) = image.shape[:2]
    (origH,origW) = (H,W)
    
    (newW, newH) = (320,320)
    rW = W / float(newW)
    rH = H / float(newH)
    
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
    
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            
            (offsetX, offsetY) = (x * 4.0, y * 4.0) #since o/p of conv network is 1/4th of original size
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    return (rects, confidences),rW,rH,orig,(origH,origW)
            
   
  



    
    
    

