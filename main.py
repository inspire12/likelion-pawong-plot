from fastapi import FastAPI, Path
from service.pca_service import PCAService
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/plot/2d")
async def get_pca():
    p = PCAService()
    p.visualize_2d()
    return {"message": "Hello World"}

@app.get("/plot/3d")
async def get_pca():
    p = PCAService()
    p.visualize_3d()
    return {"message": "Hello World"}
