from sys import prefix
import image_classifier_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(image_classifier_router.router, prefix="/Image")

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'dummy check! classifier  is all ready to go!'