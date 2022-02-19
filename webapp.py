from wsgiref import headers
import streamlit as st
from PIL import Image, ExifTags
import io
import os
import requests

def exifCheck(bytes_data):
    try:
        image = Image.open(io.BytesIO(bytes_data))

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = image._getexif()

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)


    except (AttributeError, KeyError, IndexError):
    # cases: image don't have getexif
        pass
    return image

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

st.write("""
# MuseumAI
In questa **webapp** è possibile selezionare una foto scattata ad un **oggetto** durante la visita al museo
e ottenere una descrizione fornita dall'intelligenza artificiale ***MuseumAI***
""")
uploaded_file = st.file_uploader("Scegli una foto")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image = exifCheck(bytes_data)
    st.image(image, caption='Immagine preprocessata')
    image.save("selected_img.jpg")
    
    url = "http://127.0.0.1:8000/post_images/"

    payload={}
    files=[
    ('file',('selected_img.jpg',open('selected_img.jpg','rb'),'application/octet-stream'))
    ]
    headers = {
    'accept': 'application/json',
    #'Content-Type': 'multipart/form-data'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    url_get = "http://127.0.0.1:8000/get_images/"
    
    response_get = requests.request("GET", url_get)

    img = Image.open(io.BytesIO(response_get.content))
    st.image(img, caption='processata')
    print(response.text)
















hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
