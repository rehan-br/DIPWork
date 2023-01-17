import numpy as np
import cv2

image_path = 'images/random.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/*whatever*.caffemodel'
min_confidence = 0.3


class = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

np.random.seed(536420) #Used to keep consistent colors for the classes
colors = np.random.uniform(0, 255, size=(len(class), 3)) # random colors for each class but stays consistent due to seed -- 3 means RGB

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
image = cv2.imread(image_path)

height, width = image.shape[0], image.shape[1]

#Binary Large Object
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 123) #Second and Fourth values are experimental

net.setInput(blob) 
detected_object = net.forward()

print(detected_object[0][0][0]) #First Object Detected
print(detected_object[0][0][1]) #Second Object Detected

#Second value in the printed statement will indicate the class number i.e 3 for bird
#Third value in the printed statement will indicate the confidence
#Rest of the values are the coordinates which have to be multiplied by the image height and the image width - helps in the construction of the bounding box

for i in range(detected_object.shape[2]):

    confidence = detected_object[0][0][i][2] #int not used since that would round the confidence down to 0 (Whole Number)

    if confidence > min_confidence :
        
        #Here the rest of the values are used to find the corners of the box
        class_index = int(detected_object[0, 0, i, 1])
        upperleftx = int(detected_object[0, 0, i, 3] * width)
        upperlefty = int(detected_object[0, 0, i, 4] * height)
        lowerrightx = int(detected_object[0, 0, i, 5] * width)
        lowerrighty = int(detected_object[0, 0, i, 6] * height)

        prediction_text = f"{classes[class_index]} : {confidence:.2f}%"
        cv2.rectangle(image, (upperleftx, upperlefty), (lowerrightx, lowerrighty), colors[class_index], 3) #Sets the rectangle for the found object
        cv2.putText(image, prediction_text, (upperleftx, upperlefty - 15 if upperlefty > 30 else upperlefty + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_index], 2) #Sets the text for the found object

        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

