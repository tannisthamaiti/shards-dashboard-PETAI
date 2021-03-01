from fastapi import APIRouter,File, UploadFile, Form
from model import FMIModel
from starlette.responses import StreamingResponse, JSONResponse
import numpy as np
import io

router = APIRouter()
model = FMIModel()

#read npz file
def read_imagefile(file):
    arr = np.load(io.BytesIO(file))['data']
    return arr

@router.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    arr = read_imagefile(await file.read())
    return JSONResponse(model.predict(arr))
   
@router.post("/predict1")
async def train_image(maxiters: str = Form(...)):
    return JSONResponse(model.predict1(int(maxiters)))
