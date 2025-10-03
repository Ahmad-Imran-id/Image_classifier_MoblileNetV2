import numpy as np 
import cv2
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import(
MobileNetV2,
preprocess_input,
decode_predictions
)

from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess(image):
    img=np.array(image)
    img=cv2.resize(img,(224,224))
    img=preprocess_input(img)
    img=np.expand_dims(img,axis=0)
    return img


def classify(model,image):
    proc_img=preprocess(image)
    predictions=model.predict(proc_img)
    decoded_predictions = decode_predictions(predictions,top=3)[0]
    return decoded_predictions

def main():
    st.set_page_config(page_title='Image Classifier',page_icon='🖼️')
    st.title('AI Image classifier')
    st.write('Uplaod an image to see what AI thinks it is!')

    @st.cache_resource()
    def load_cashed_model():
        return load_model()
    
    model=load_cashed_model()
    
    uploaded_file = st.file_uploader("Upload an image here 🖼️",type=['jpg','png'])

    if uploaded_file is not None:
        image= st.image(uploaded_file,use_column_width=True)
        btn=st.button("Classify image")
        if btn:
            with st.spinner('Analyzing Image'):
                image=Image.open(uploaded_file)
                predictions=classify(model,image)
                
                if predictions:
                    st.subheader('Predictions')
                    for _,label,score in predictions:
                        st.write(f'**{label}** : {score:.2%}')

if __name__=='__main__':
    main()





