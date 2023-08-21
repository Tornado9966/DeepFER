from __future__ import print_function
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from deep_emotion import FERModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

# Use pretrained model
net = FERModel()
net.load_state_dict(torch.load('fermodel.pt'))
net.to(device)
net.eval()

def load_img(path):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)

emotions = {
    0: ['Angry', (0,0,255), (255,255,255)],
    1: ['Disgust', (0,102,0), (255,255,255)],
    2: ['Fear', (255,255,153), (0,51,51)],
    3: ['Happy', (153,0,153), (255,255,255)],
    4: ['Sad', (255,0,0), (255,255,255)],
    5: ['Surprise', (0,255,0), (255,255,255)],
    6: ['Neutral', (160,160,160), (255,255,255)]
}

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def processImage(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    counter = 0
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        counter+=1
        roi = img[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi,(48,48))
        cv2.imwrite("roi" + str(counter) + ".jpg", roi)

        imgg = load_img("roi" + str(counter) + ".jpg")
        out = net(imgg)
        pred = F.softmax(out, dim=1)
        classs = torch.argmax(pred,1)
        prediction = emotions[classs.item()][0]

        cv2.rectangle(img, (x, y), (x+w, y+h), emotions[classs.item()][1], 2, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x, y-20), (x+w, y), emotions[classs.item()][1], -1, lineType=cv2.LINE_AA)
        cv2.putText(img, prediction, (x, y-5), 0, 0.6, emotions[classs.item()][2], 2, lineType=cv2.LINE_AA)
    
    return img

def visualize(path):
    # Read the image
    img = cv2.imread(path)
    result = processImage(img)
    cv2.imwrite("result.jpg", result)
    plt.imshow(result)

def visualizeVideo(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_h = 360
    target_w = int(target_h * frame_width / frame_height)
    out = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),
                          fps, (target_w,target_h))
    
    while True:
        success, image = cap.read()
        if success:
            result = processImage(image)
            out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
