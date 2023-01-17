import cv2

# Load a pre-trained object detection model
net = cv2.dnn.readNetFromCaffe("path/to/model.prototxt", "path/to/weights.caffemodel")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Get a frame from the webcam
    _, frame = cap.read()

    # Resize the frame to 300x300
    frame = cv2.resize(frame, (300, 300))

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only show detections with a confidence greater than 0.5
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()