
import joblib
import cv2
import numpy as np
from PIL import Image
from face_recognition import preprocessing
#from .util import draw_bb_on_img
from inference import util
#import util
#from .constants import MODEL_PATH
from fastapi import FastAPI
app = FastAPI()


@app.get("/")
async def root():
    print("running")
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    face_recogniser = joblib.load('model/face_recogniser.pkl')
    preprocess = preprocessing.ExifOrientationNormalize()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img = Image.fromarray(frame)
        faces = face_recogniser(preprocess(img))
        if faces is not None:
            util.draw_bb_on_img(faces, img)

        # Display the resulting frame
        cv2.imshow('video', np.array(img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    cv2.destroyAllWindows()
