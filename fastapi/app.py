from sys import prefix
import image_classifier_router
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(image_classifier_router.router, prefix="/Image")

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'dummy check! classifier  is all ready to go!'

@app.get("/image_return/{file}")
async def img_return(file :str):
    ## add the filename only to get the data as image format
    return FileResponse("outputs/"+file+".png")