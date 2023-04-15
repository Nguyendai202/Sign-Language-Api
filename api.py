import io
import cv2
import numpy as np
from keras.models import load_model
from keras import models
from fastapi import FastAPI,File,UploadFile
from pydantic import BaseModel
import tensorflow as tf

path_model = "model/sign-language.h5"
model = tf.keras.models.load_model(path_model )
# model = models.load_model(path_model)
app = FastAPI()
class PredictionResult(BaseModel):
    prediction: str  #định nghĩa cấu trúc data trả về 
@app.post("/predict")
async def predict_sign_langue(file: UploadFile= File(...))->PredictionResult: # chỉ định kiểu dữ liệu trả về 
    contents = await file.read()# đọc dạng byte
    nparr = np.frombuffer(contents,np.uint8)# chuyển sang numpy 
    img  = cv2.imdecode(nparr,cv2.IMREAD_COLOR)# giải mã ảnh > đối tượng numpy có màu sắc 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    crop_resize =cv2.resize(img,(28,28)) 
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
        return PredictionResult(prediction=letter)
    else:
        return PredictionResult(prediction="No prediction")
if __name__ == "__main__":
    import uvicorn
    print("In processing...")
    uvicorn.run(app= app, host= "127.0.0.1", port= 8000)
    print("Shutting down...")

    

