# python classify_video.py --model split20.model --labelbin split20.pickle --input path_to_input_video.avi --output path_to_output.avi

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
from copy import deepcopy
import numpy as np
import argparse
import time
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-s", "--skip-frames", type=int, default=1,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

writer = None
W = None
H = None

totalFrames = 0
fps = FPS().start()

while True:
    current_frame = vs.read()
    current_frame = current_frame[1] if args.get("input", False) else current_frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and current_frame is None:
        break
    current_frame = imutils.resize(current_frame, width=1920)
    if W is None or H is None:
        (H, W) = current_frame.shape[:2]

    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args["output"], fourcc, 240,
            (W,H), True)

    if totalFrames % args["skip_frames"] == 0:
        frame = deepcopy(current_frame)
        # pre-process the image for classification
        frame = cv2.resize(frame, (96, 96))
        frame = frame.astype("float") / 255.0
        frame = img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)

        # classify the input image
        proba = model.predict(frame)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        # build the label and draw the label on the image
        floor_label = "Floor Capacity: {}".format(label)
        accuracy_label = "Accuracy:       {:.2f}".format(proba[idx] * 100)
        cv2.putText(current_frame, floor_label, (15, 42),  cv2.FONT_HERSHEY_SIMPLEX,
            1.7, (0, 0, 0), 4)
        cv2.putText(current_frame, accuracy_label, (15, 95),  cv2.FONT_HERSHEY_SIMPLEX,
            1.7, (0, 0, 0), 4)

    if writer is not None:
        writer.write(current_frame)

    cv2.imshow("Frame", current_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop	
    if key == ord("q"):
        break
    totalFrames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
