
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import os
from random import randint
import uuid
import numpy
from numpy.lib.function_base import copy

from numpy.lib.type_check import imag
from starlette.responses import FileResponse
from utils import create_initial_cfgs, load_cfgs, feed_forward, load_image_into_numpy_array
from pydantic import BaseModel
import aiofiles

app = FastAPI()

image_name = "destination.jpg"

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
modelli = load_cfgs(create_initial_cfgs(),'3.0')

@app.post("/post_images/")
async def create_upload_file(file: UploadFile = File(...)):

    image = await file.read()
    
    async with aiofiles.open(image_name , "wb") as f:
        await f.write(image)
    
    feed_forward(image_name, modelli)
    
    return FileResponse(image_name, media_type="image/jpg")


@app.get("/get_images/")
async def read_random_file():

    return FileResponse(image_name)

class City(BaseModel):
    name: str

@app.post("/test_post")
def create_city(city: City):
    return city
