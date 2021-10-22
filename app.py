import streamlit as st
import os
from PIL import Image, ImageOps
import utils

st.title("Image Super Resolution")
st.text("Upload the image here")


#creating file uploader
image_file = st.file_uploader(label = '', type = ['png', 'jpg', 'jpeg'])

INPUT_PATH = "./input.png"
OUT_PATH = "./output.png"

if image_file is None:
    try:
        os.remove(INPUT_PATH)
        os.remove(OUT_PATH)

    except FileNotFoundError:
        st.success("No previous images.")

else:
    in_img = Image.open(image_file)
    
    #saving the input image in my local disk
    with open(INPUT_PATH, "wb") as file:
        file.write(image_file.getbuffer())



if st.button("Process"):
    st.header("Low Resolution Image")
    st.image(in_img, width = 300)

    out_img = utils.get_prediction(INPUT_PATH)
    out_img.save(OUT_PATH)

    st.header("High Resolution Image:")
    st.image(out_img, width = 300)


    

