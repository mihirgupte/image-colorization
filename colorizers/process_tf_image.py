from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import cv2
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

splitting_count = 1

def from_lab_to_rgb(gray_images,ab_images,n=10):
    Zeros_Imp = np.zeros((n,224,224,3))
    
    Zeros_Imp[:,:,:,0] = gray_images[0:n:]
    Zeros_Imp[:,:,:,1:] = ab_images[0:n:]
    
    Zeros_Imp = Zeros_Imp.astype("uint8")
    
    Main_Img = []
    
    for indexing in range(0,n):
        Main_Img.append(cv2.cvtColor(Zeros_Imp[indexing],cv2.COLOR_LAB2RGB))
        
    Main_Img = np.array(Main_Img)
    
    return Main_Img

def line_image(gray_images,splitting_count=splitting_count,preprocess_function=preprocess_input):
    Zeros_Imp = np.zeros((splitting_count,224,224,3))
    #print(gray_images[:1])
    for indexing in range(0,3):
        Zeros_Imp[:splitting_count,:,:,indexing] = gray_images[:splitting_count]
        
    return preprocess_function(Zeros_Imp)

def process_img(img):
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = line_image(np.expand_dims(img,0),splitting_count)
    #X = preprocess_input(img)s
    #print(img)
    return img#np.expand_dims(X,0)

def predict_and_output(img, model):
    dims = (300,300)
    img_new = model.predict(img)
    img_new = cv2.resize(np.squeeze(img_new,0), dims, interpolation = cv2.INTER_AREA)
    return img_new
    # cv2.imshow('img_new',img_new)
    # cv2.waitKey()

# Gray_NPY = np.load("./data/gray_scale.npy")
# cv2.imshow('img',Gray_NPY[1])
# cv2.waitKey()

# img = Gray_NPY[1]
# cv2.imwrite('./imgs/dataset.jpg',img)
# img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# img = process_img(img)
# colorization = tf.keras.models.load_model('./data/colorization (1).h5')
# img = predict_and_output(img)
# # img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
# cv2.imshow('img_new',img)
# cv2.waitKey()