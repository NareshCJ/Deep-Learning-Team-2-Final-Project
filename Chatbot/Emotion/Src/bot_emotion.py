import numpy as np
import cv2
import torch
import torchvision
import tarfile
import torch.nn as nn
import torchvision.transforms as T

from classes import ResNet_expression



face_cascade = cv2.CascadeClassifier('/home/praveen/Desktop/Projects/technocolab_project_2/emotion_recognition/src/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

Labels = {
    0:'Angry',
    1:'Disgust',
    2:'Fear',
    3:'Happy',
    4:'Sad',
    5:'Surprise',
    6:'Neutral'
}

stats = ([0.5],[0.5])
tsfm = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(), 
    T.Normalize(*stats,inplace=True)
])

device = torch.device('cuda')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    #xb = img.unsqueeze(0)
    yb = model(img)
    _, preds  = torch.max(yb, dim=1)
    return Labels[preds[0].item()]

input_size = 48*48
output_size = 7

model2 = to_device(ResNet_expression(1, output_size), device)

model2.load_state_dict(torch.load('/home/praveen/Desktop/Projects/technocolab_project_2/emotion_recognition/models/resnet-facial.pth'))


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        gray = gray.astype(np.uint8)
        gray = cv2.resize(roi_gray, (48, 48))
        transformed_image = tsfm(gray)
        transformed_image = transformed_image.to(device)
        #print(transformed_image.unsqueeze(0))
        print(' Predicted:', predict_image(transformed_image.unsqueeze(0), model2))

        
        

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
