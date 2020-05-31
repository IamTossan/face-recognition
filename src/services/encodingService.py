import os
import json
import time
import pickle

KNOWN_FACES_DIR = 'known_faces'

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