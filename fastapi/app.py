from sys import prefix
from fastapi import FastAPI
import image_classifier_router

app = FastAPI()
app.include_router(image_classifier_router.router, prefix="/Image")

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'dummy check! classifier  is all ready to go!'