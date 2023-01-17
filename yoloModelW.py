import cv2

# Load the YOLOv3 model
net = cv2.dnn.readNet("path/to/yolov3.weights", "path/to/yolov3.cfg")
classes = []
with open("path/to/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Get a frame from the webcam
    _, frame = cap.read()

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (41616,41616), (0, 0, 0), True, crop=False)

    # Pass the blob through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize the list of detected objects
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get the bounding box
                x, y, w, h = detection[0:4]
                # Scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                # Draw the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the detections on the frame
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, classes[class_ids[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Break the loop
