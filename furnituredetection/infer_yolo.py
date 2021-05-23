#/usr/bin/python

import os
import cv2
import numpy as np



if __name__ == '__main__':

	#Paths to YOLOv4 tiny weights and cfg files.
	weights_path = './yolov3-tiny-obj_best.weights' 
	config_path = './yolov3-tiny-obj.cfg'
	labels_path = './obj.names'

	#Load model based on cfg files and trained weights
	model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
	ln = model.getLayerNames()
	#Get output layers
	ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]

	#Read labels from .names file
	labels = open(labels_path).read().strip().split("\n")

	color = (0,255,0)

	for img_name in os.listdir('test'):
		#read images in test folder
	    img_path = os.path.join('test', img_name)
	    image = cv2.imread(img_path)
	    
	    #Get image size, reshape and normalize.
	    (H, W) = image.shape[:2]
	    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

	    #Run image through model.
	    model.setInput(blob)
	    layerOutputs = model.forward(ln)
	    
	    
	    # Initializing for getting box coordinates, confidences, classid 
	    boxes = []
	    confidences = []
	    classIDs = []
	    threshold = 0.01
	    
	    for output in layerOutputs:
	        for detection in output:
	            scores = detection[5:]
	            classID = np.argmax(scores)
	            confidence = scores[classID]
	            if confidence > threshold:
	                box = detection[0:4] * np.array([W, H, W, H])
	                (centerX, centerY, width, height) = box.astype("int")           
	                x = int(centerX - (width / 2))
	                y = int(centerY - (height / 2))    
	                boxes.append([x, y, int(width), int(height)])
	                confidences.append(float(confidence))
	                classIDs.append(classID)
	    
	    #non-maxima suppression
	    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
	    
	    if len(idxs) > 0:
	        for i in idxs.flatten():
	            (x, y) = (boxes[i][0], boxes[i][1])
	            (w, h) = (boxes[i][2], boxes[i][3])
	            
	            lbl = labels[classIDs[i]]
	            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	            cv2.putText(image, '%s' % lbl, (x, y - 10),                     
	            cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	    
	    out_name = img_name.split('.')[0] + '_out.' + 'jpg'
	    out_path = os.path.join('output', out_name)
	    cv2.imwrite(out_path, image)