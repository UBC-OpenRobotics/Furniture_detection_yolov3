import os
import cv2
import numpy as np
import json

# Takes in the path of target image and returns class ids and bounding box labels as JSON strings
# Return format [[c1,x1,y1,w1,h1],[c2,x2,y2,w2,h2]...]. Returns [] if no detection.

def get_detections(path):
    weights_path = './yolov3-tiny-obj_best.weights' 
    config_path = './yolov3-tiny-obj.cfg'
    #Load model based on cfg files and trained weights
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = model.getLayerNames()
    #Get output layers
    ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    image = cv2.imread(path)

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
    threshold = 0.15

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
    results = []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
    if len(idxs) > 0:
	    for i in idxs.flatten():
	        (x, y) = (boxes[i][0], boxes[i][1])
	        (w, h) = (boxes[i][2], boxes[i][3])
	        results.append([int(classIDs[i]),int(x),int(y),int(w),int(h)])  
    results_json = json.dumps(results)
    return results_json


# Testing if above function is working

path = './test/five.jpg'
detections_json = get_detections(path)
detections = json.loads(detections_json)
label_map = {0: 'Chair', 1:'Sofa', 2:'Table'}
image = cv2.imread(path)
for d in detections:
    cv2.rectangle(image, (d[1], d[2]), (d[1] + d[3], d[2] + d[4]), (0,255,0), 2)
    cv2.putText(image, '%s' % label_map[d[0]], (d[1], d[2] - 10),                     
    cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0), 2)

cv2.imshow('Results',image)
cv2.waitKey(0)


    

    