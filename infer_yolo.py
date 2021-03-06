#/usr/bin/python

import os
import cv2
import numpy as np
import argparse as ap

class Model():

    def __init__(self):

        #Define Paths to YOLOv4 tiny weights and cfg files
        self.weights_path = './model_data/yolov4-tiny-bags_best.weights' 
        self.config_path = './model_data/yolov4-tiny-bags.cfg'
        self.labels_path = './model_data/bags.names'

        #Visual settings
        self.color = (0,255,0)

        #Lables
        self.labels = self.read_labels(self.labels_path)

        #Load saved model
        self.model,self.ln = self.load_model()

    def load_model(self):
        #Load model based on cfg files and trained weights
        model = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        ln = model.getLayerNames()
        #Get output layers
        ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]

        return model, ln

    def read_labels(self, labels_path):
        return open(labels_path).read().strip().split("\n")

    def infer(self, image):
        #Get image size, reshape and normalize.
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        #Run image through model.
        self.model.setInput(blob)
        layerOutputs = self.model.forward(self.ln)
        
        
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
        
        #non-maxima suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                lbl = self.labels[classIDs[i]]
                bbox = [x,y,w,h]

        return lbl, bbox
    
    def draw_bbox(self, image, lbl, bbox):
        
        x,y,w,h = bbox
        show_img = image.copy()
        cv2.rectangle(show_img, (x, y), (x + w, y + h), self.color, 2)
        cv2.putText(show_img, '%s' % lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, self.color, 2)
        
        return show_img





if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-i','--input',required=True, help='Path to image or directory of images')
    parser.add_argument('-s','--show', required=False, action='store_true')
    args = parser.parse_args()

    #Process image path or directory
    input_path = args.input
    if os.path.isdir(input_path):
        img_paths = [os.path.join(input_path, filename) for filename in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        img_paths = [input_path]
    else:
        print('[ERROR] %s is not a directory or file' % input_path)
        exit()
        

    #Load model
    model = Model()



    for img_path in img_paths:
        #load image
        image = cv2.imread(img_path)

        #run inference
        lbl, bbox = model.infer(image)

        print('%s: Label: %s\tBbox: (%i,%i,%i,%i)' % (img_path, lbl, bbox[0],bbox[1],bbox[2],bbox[3]))

        if args.show:
            show_img = model.draw_bbox(image, lbl, bbox)
            cv2.imshow('Results', show_img)
            cv2.waitKey(-1)

    if args.show:
        cv2.destroyAllWindows()