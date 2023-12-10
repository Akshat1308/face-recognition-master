from collections import namedtuple
from datetime import datetime
import pandas as pd

import csv

Prediction = namedtuple('Prediction', 'label confidence')
Face = namedtuple('Face', 'top_prediction bb all_predictions')
BoundingBox = namedtuple('BoundingBox', 'left top right bottom')

attendance = []



now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
# print(current_date)

 


def top_prediction(idx_to_class, probs):
    top_label = probs.argmax()

    # if probs[top_label] > 0.99:
        # now = datetime.now()
        # current_date = now.strftime("%Y-%m-%d")
        # print(idx_to_class[top_label], probs[top_label])
        # attendance[idx_to_class[top_label]] = ['P', probs[top_label], current_date]
        # lnwriter = csv.writer(f)
        # lnwriter.writerow(attendance)
    return Prediction(label=idx_to_class[top_label], confidence=probs[top_label])


def to_predictions(idx_to_class, probs):
    return [Prediction(label=idx_to_class[i], confidence=prob) for i, prob in enumerate(probs)]


class FaceRecogniser:
    def __init__(self, feature_extractor, classifier, idx_to_class):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.idx_to_class = idx_to_class

    def recognise_faces(self, img):
        bbs, embeddings = self.feature_extractor(img)
        # print(embeddings)
        if bbs is None:
            # if no faces are detected
            return []
        predictions = self.classifier.predict_proba(embeddings)
        # bb=BoundingBox(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3])
        for probs in predictions:
            pred = top_prediction(self.idx_to_class, probs)

        if pred.label not in attendance and pred.confidence > 0.9:
            attendance.append(pred.label)
            f= open(current_date+'.csv','a+',newline = '')
            lnwriter = csv.writer(f) 
            lnwriter.writerow([pred.label, pred.confidence, current_date])

    
        # print(pred)
        
        
        
        # l = []
        # for probs in predictions:
        #     pred = top_prediction(self.idx_to_class, probs)
        #     if pred.confidence > 0.90:
        #         l.append(pred)
        #         print(pred)
        # l=set(l)
        # attendance = {'name':l}
        # print(attendance)
        
        return [
            Face(
                top_prediction=top_prediction(self.idx_to_class, probs),
                bb=BoundingBox(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3]),
                all_predictions=to_predictions(self.idx_to_class, probs)
            )
            for bb, probs in zip(bbs, predictions)
        ]

        
    def __call__(self, img):
        return self.recognise_faces(img)
