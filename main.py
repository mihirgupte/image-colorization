from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from colorizer import image_colorizer
from PIL import Image
# from mega import Mega
from colorizers import *
from colorizer import *
import cv2
import io
import base64

app = FastAPI()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
colorizer_eccv16 = eccv16(pretrained=True).eval()
# mega = Mega()
# m = mega.login()
# try:
#     m.download_url('https://mega.nz/file/EMl2FAqR#f54U3M3-s7eMz-YAnsGvzqp1NsJkJme74UT0Tf9_Haw',"./model/")
# except PermissionError:
#     print("File is being used but that's okay since the file might be already downloaded")

class UserIn(BaseModel):
    imgObject: str

@app.get("/")
async def test_if_working():
    return "HELLO WORLD"

@app.post("/colorized", response_model=UserIn)
def colorized_image(user: UserIn):
    img = data_uri_to_cv2_img(user.imgObject)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    base64_img = base64.b64encode(out_img_eccv16)
    cv2.imshow('image',out_img_eccv16)
    cv2.waitKey(0)
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    base64_img = base64.b64encode(out_img_siggraph17)
    cv2.imshow('image',out_img_siggraph17)
    cv2.waitKey(0)
    return {"imgObject": base64_img}