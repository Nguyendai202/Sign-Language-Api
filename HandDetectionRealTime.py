import cv2
import numpy as np 
from keras import models 
path_model = "model/sign-language.h5"
model =  models.load_model(path_model)
vid =  cv2.VideoCapture(0)
while True:
    ret,frame = vid.read()
    cv2.rectangle(frame,(100,50),(600,600),(255,0,0),2)
    crop = frame[80:400,80:400] # Khung hình chứa đối tượng 
    crop_resize =cv2.resize(crop,(28,28)) 
    img_read = (cv2.cvtColor(crop_resize,cv2.COLOR_RGB2GRAY))/255.0
    data = img_read.reshape(-1,28,28,1)
    model_output = model.predict([data])[0]
    label = np.argmax(model_output)
    if max(model_output)>0.8:
        if label==0:
            letter = "A"
        elif label ==1:
            letter = "B"
        elif label == 2:
            letter = "C"
        elif label == 3:
            letter = "D"
        elif label == 4:
            letter = "E"
        elif label == 5:
            letter = "F"
        elif label == 6:
            letter = "G"
        elif label == 7:
            letter = "H"
        elif label == 8:
            letter = "I"
        elif label == 10:
            letter = "K"
        elif label == 11:
            letter = "L"
        elif label == 12:
            letter = "M"
        elif label == 13:
            letter = "N"
        elif label == 14:
            letter = "O"
        elif label == 15:
            letter = "P"
        elif label == 16:
            letter = "Q"
        elif label == 17:
            letter = "R"
        elif label == 18:
            letter = "S"
        elif label == 19:
            letter = "T"
        elif label == 20:
            letter = "U"
        elif label == 21:
            letter = "V"
        elif label == 22:
            letter = "W"
        elif label == 23:
            letter = "X"
        elif label == 24:
            letter = "Y"
        print(letter)
        cv2.putText(frame,letter,(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break 
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
if __name__ == "__main__":
    path_model = "model/sign-language.h5"
    model =  models.load_model(path_model)
        

