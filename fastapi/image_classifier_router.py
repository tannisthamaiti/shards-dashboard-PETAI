from fastapi import APIRouter,File, UploadFile
from model import FMIModel
from starlette.responses import StreamingResponse, JSONResponse
import numpy as np
import io

router = APIRouter()
 
#read npz file
def read_imagefile(file):
    arr = np.load(io.BytesIO(file))['data']
    return arr

@router.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    model = FMIModel()
    arr = read_imagefile(await file.read())
    return JSONResponse(model.predict(arr))