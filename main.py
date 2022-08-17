import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import seaborn as sns
import base64
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

model = keras.models.load_model("./model/model_unet_no_aug.h5")

# categories = {
#     'void': [0, 1, 2, 3, 4, 5, 6],
#     'flat': [7, 8, 9, 10],
#     'construction': [11, 12, 13, 14, 15, 16],
#     'object': [17, 18, 19, 20],
#     'nature': [21, 22],
#     'sky': [23],
#     'human': [24, 25],
#     'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
# }
categories = {
    'void': [0],
    'flat': [1],
    'construction': [2],
    'object': [3],
    'nature': [4],
    'sky': [5],
    'human': [6],
    'vehicle': [7]
}

def convertCategories(x):
    if x in categories['void']:
        return 0
    elif x in categories['flat']:
        return 1
    elif x in categories['construction']:
        return 2
    elif x in categories['object']:
        return 3
    elif x in categories['nature']:
        return 4
    elif x in categories['sky']:
        return 5
    elif x in categories['human']:
        return 6
    elif x in categories['vehicle']:
        return 7

convertCategories_v = np.vectorize(convertCategories)

def get_cat_name(cat):
    if cat == 0:
        return "void"
    elif cat == 1:
        return "flat"
    elif cat == 2:
        return "construction"
    elif cat == 3:
        return "object"
    elif cat == 4:
        return "nature"
    elif cat == 5:
        return "sky"
    elif cat == 6:
        return "human"
    elif cat == 7:
        return "vehicle"
    
def describe_mask_content(mask):
    msk_cats = np.unique(mask)
    converted_cats = []
    for cat in msk_cats:
        # converted_cats.append(convertCategories(cat))
        id = convertCategories(cat)
        converted_cats.append(get_cat_name(id))
    return np.unique(converted_cats)

def preprocessImg(img):
    img_matrix = np.expand_dims(img, 2)
    converted_img = convertCategories_v(img_matrix)
    return converted_img

palette = sns.color_palette("turbo", len(categories))

def set_mask_color(mask):
    cats_colors = palette
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    # Masque RVB (3 canaux)
    mask_img = np.zeros((mask.shape[0], mask.shape[1], 3)).astype('float')
    for color_num in range(8):
        # Le numéro de canal est le même que l'index des palette de couleurs
        mask_color = (mask == color_num)
        for i in range(3):
            # valeurs de i :
            # 0: rouge
            # 1: vert
            # 2: bleu
            mask_img[:,:,i] += (mask_color*(cats_colors[color_num][i]))
    return(mask_img)

def resize_img(img, width=128, height=128, interpolation=None):
    if interpolation == None:
        resized_img = cv2.resize(img, (width,height))
    else:
        resized_img = cv2.resize(img, (width,height), interpolation=interpolation)
    return resized_img

def segment_it(img):
    y_pred = model.predict(img)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    return y_pred_argmax

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/")
async def get_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # ici, l'image est décodée
    img_dimensions = img.shape
    img_ = resize_img(img, 128, 128)
    img_ = image.img_to_array(img_)/255.
    conv_img = np.empty(
        (
            1, 
            *(128,128), 
            3
        )
    )
    conv_img[0,] = img_
    
    # TODO implémenter prédiction
    
    # y_pred = img # just for a test
    y_pred = segment_it(conv_img)
    msk = y_pred[0]
    msk_content = str(describe_mask_content(msk))
    msk = preprocessImg(msk)[:,:,0]
    msk = set_mask_color(msk)
    msk = np.multiply(msk, 255.)
    msk = resize_img(msk, img_dimensions[1], img_dimensions[0], interpolation=cv2.INTER_CUBIC)
    overlay = cv2.addWeighted(msk.astype(np.float64), 0.5, img.astype(np.float64), 0.5, 0 )
    # y_pred = resize_img(img)

    # _, encoded_img = cv2.imencode('.PNG', msk) # là, elle est réencodée
    _, encoded_img = cv2.imencode('.PNG', overlay)

    encoded_img = base64.b64encode(encoded_img)
    
    return{
        'filename': file.filename,
        'dimensions': img_dimensions,
        'encoded_img': encoded_img,
        'test_size': msk.shape,
        "np_unique": list(np.unique(msk)),
        "detected_objects": msk_content
    }
