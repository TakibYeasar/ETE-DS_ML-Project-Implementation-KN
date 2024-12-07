from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from api.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Route for the home page


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predictdata", response_class=HTMLResponse)
async def predict_data_get(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predictdata", response_class=HTMLResponse)
async def predict_data_post(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("After Prediction")
    return templates.TemplateResponse("home.html", {"request": request, "results": results[0]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
