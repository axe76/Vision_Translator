# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:45:37 2020

@author: ACER
"""

from text_detect import detect
import pytesseract
import numpy as np
from imutils.object_detection import non_max_suppression

def ocr(path):
    (rects, confidences),rW,rH,orig,(origH,origW) = detect(path)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    
    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        padding = 0.0
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        
        roi = orig[startY:endY, startX:endX]
        config = ("-l deu --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        results.append(text)
        #results.append(((startX, startY, endX, endY), text))
        
    #print(results)
    return results
	







    
    
    

