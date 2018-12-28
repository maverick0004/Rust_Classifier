
import numpy as np
import os
import sys
import keras as K
import json
import base64
from io import BytesIO
from PIL import Image
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input

def init():
    
    global model  

    print("Executing init() method...")
    print("Python version: " + str(sys.version) + ", keras version: " + K.__version__)
    # Load the model 
    model = K.models.load_model('azureml-models/kerasmodel/8/kerasmodel.pkl')
    #model = K.models.load_model('kerasmodel.pkl')
    
    return

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict_classes(x)
    return preds[0]

def run(inputString):
    
    responses = []
    base64Dict = json.loads(inputString)

    for k, v in base64Dict.items():
        img_file_name, base64Img = k, v
    decoded_img = base64.b64decode(base64Img)
    img_buffer = BytesIO(decoded_img)
    #imageData = Image.open(img_buffer).convert("RGB")


#     # Evaluate the model using the input data
#     img = ImageOps.fit(imageData, (32, 32), Image.ANTIALIAS)
#     img_conv = np.array(img) # shape: (32, 32, 3)
#     # Scale pixel intensity
#     x_test = img_conv / 255.0
#     # Reshape
#     x_test = np.moveaxis(x_test, -1, 0)
#     x_test = np.expand_dims(x_test, 0)  # shape (1, 3, 32, 32)

#     y_pred = model.predict(x_test)
#     y_pred = np.argmax(y_pred, axis=-1)
#     # print(y_pred)
#     data = json.loads(inputString)['data']
    #img_input = base64.decodestring(json.dumps(inputString)[0])
    img = image.load_img(img_buffer,target_size=(224, 224))
    #img = json.loads(inputString)['data']
    preds = predict(model, img)
    LABELS = ["Non Rust", "Rust"]
    resp = LABELS[preds]
    responses.append(resp)
    return json.dumps(responses)
    
  
if __name__ == "__main__":
    init()
    # input data
    img_path = '00000668.JPG'
    #input_data = "{\"data\": [" + str(list(img_path)) + "]}"
    encoded = None
    with open(img_path, 'rb') as file:
        encoded = base64.b64encode(file.read())
    img_dict = {img_path: encoded.decode('utf-8')}
    body = json.dumps(img_dict)
    resp = run(body)
    print(resp)