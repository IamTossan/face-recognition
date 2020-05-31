#!/usr/bin/env python3

import os
import time
import json
import cv2
import face_recognition
import pickle

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
MODEL = 'cnn'
OPTIMIZATION_FACTOR = 4

next_id = 0

def getRegisteredEncodings():
    known_faces = []
    known_names = []

    ids = None
    with open('ids.json') as f:
        ids = json.load(f)

    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
            filePath = f'{KNOWN_FACES_DIR}/{name}/{filename}'
            print(f'loading {filePath}')

            encoding = pickle.load(open(filePath, 'rb'))
            known_faces.append(encoding)
            if ids and name in ids:
                known_names.append(ids[name])
            else:
                known_names.append(name)

    return known_names, known_faces

def registerEncoding(name, encoding):
    path = f'{KNOWN_FACES_DIR}/{name}'
    os.mkdir(path)
    pickle.dump(encoding, open(f'{path}/{name}-{int(time.time())}.pkl', 'wb'))

def drawFaceIdentifications(frame, faces):
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

        top *= OPTIMIZATION_FACTOR
        right *= OPTIMIZATION_FACTOR
        bottom *= OPTIMIZATION_FACTOR
        left *= OPTIMIZATION_FACTOR

        color = (0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_thickness = 2
        font_thickness = 1

        cv2.rectangle(frame, (left, top), (right, bottom), color, frame_thickness)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, match, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), font_thickness)

def identify():
    video = cv2.VideoCapture(0)

    # this is for processiong identification optimization
    should_process = False

    while True:

        ret, frame = video.read()

        small_frame = cv2.resize(frame, (0, 0), fx=1/OPTIMIZATION_FACTOR, fy=1/OPTIMIZATION_FACTOR)

        if(should_process):
            locations = face_recognition.face_locations(small_frame, model=MODEL)
            encodings = face_recognition.face_encodings(small_frame, locations)

            print(f'found {len(encodings)} face(s)')
            if(len(encodings)):
                drawFaceIdentifications(frame, zip(encodings, locations))
        should_process = not should_process

        cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_names, known_faces = getRegisteredEncodings()
    next_id = len(known_names)
    identify()