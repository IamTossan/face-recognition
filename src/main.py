#!/usr/bin/env python3

import cv2
import face_recognition
from services.encodingService import getRegisteredEncodings, registerEncoding

TOLERANCE = 0.6
MODEL = 'cnn'
OPTIMIZATION_FACTOR = 4

next_id = 0

class Trail:
    def __init__(self, fn):
        self.fn = fn
        self.lastArgs = []

    def __call__(self, *args):
        if not args[1]:
            self.fn(args[0], self.lastArgs)
            self.lastArgs = []
            return None
        self.lastArgs = args[1]
        self.fn(*args)

@Trail
def draw(frame, identifications):
    for identification in identifications:
        top = identification['top'] * OPTIMIZATION_FACTOR
        right = identification['right'] * OPTIMIZATION_FACTOR
        bottom = identification['bottom'] * OPTIMIZATION_FACTOR
        left = identification['left'] * OPTIMIZATION_FACTOR
        name = identification['name']

        color = (0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_thickness = 2
        font_thickness = 1

        cv2.rectangle(frame, (left, top), (right, bottom), color, frame_thickness)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), font_thickness)

    cv2.imshow('capture', frame)

def getIdentifications(frame, faces):
    identifications = []
    for face_encoding, (top, right, bottom, left) in faces:
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'Match found: {match}')
        else:
            global next_id
            match = str(next_id)
            known_names.append(match)
            known_faces.append(face_encoding)
            registerEncoding(match, face_encoding)
            next_id += 1

        identifications.append({
            'top': top,
            'right': right,
            'bottom': bottom,
            'left': left,
            'name': match
        })

    return identifications

def run():
    video = cv2.VideoCapture(0)

    # this is for processiong identification optimization
    should_process = False

    while True:

        ret, frame = video.read()

        small_frame = cv2.resize(frame, (0, 0), fx=1/OPTIMIZATION_FACTOR, fy=1/OPTIMIZATION_FACTOR)

        identifications = []
        if(should_process):
            locations = face_recognition.face_locations(small_frame, model=MODEL)
            encodings = face_recognition.face_encodings(small_frame, locations)

            print(f'found {len(encodings)} face(s)')
            if(len(encodings)):
                identifications = getIdentifications(frame, zip(encodings, locations))
        should_process = not should_process

        draw(frame, identifications)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_names, known_faces = getRegisteredEncodings()
    next_id = len(known_names)
    run()
