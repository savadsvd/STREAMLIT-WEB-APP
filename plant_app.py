
import streamlit as st 
import tensorflow as tf 
import numpy as np
from PIL import Image,ImageOps
st.title("Plant Disease Classification")
upload_file=st.file_uploader("Upload the plant leafe",type=['jpg','png'])
generate_pred=st.button("predict")
model=tf.keras.models.load_model("/content/drive/MyDrive/new/plant.h5")
def import_n_predict(image_data,model):
  size=(224,224)
  image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
  image=np.asarray(image)
  img_reshape=img[np.newaxis,...]
  prediction=model.predict(img_reshape)
  return prediction
if generate_pred is None:
  st.text("please upload an image file")
else:
  image=Image.open(upload_file)
  with st.beta_expander('image',expanded=True):
    st.image(image,use_column_width=True)
  predictions=import_n_predict(image,model)
  class_names=['Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Early_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy', 'Apple___Cedar_apple_rust', 'Apple___healthy']
  st.title("This image most likely is {}".format(class_names[np.argmax(predictions)]))
  
