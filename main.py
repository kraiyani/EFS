
from src.components.evaluate_model import display_results
from typing import Dict
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

templates = Jinja2Templates(directory="templates")

description = """
API helps to get classify Unsecured Personal Loan Default Prediction
"""


app = FastAPI(
    title="EXUS - EFS API",
    description=description,
    version="0.0.1")


@app.get("/")
def read_root():
    
    return {"message": "Welcome!!  || /train_result  ||  /predict  || http://127.0.0.1:8000/docs "}


@app.get("/train_result")
async def get_metrics_page(request: Request):
    try:
    
        metrics_data = display_results()
        return templates.TemplateResponse("metrics_plot_template.html", {"request": request, "data": metrics_data})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An error occurred", "error": str(e)})


@app.get("/predict", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    F_1: int = Form(..., ge=462, le=731),
    F_3: int = Form(..., ge=0, le=506),
    F_4: int = Form(..., ge=0, le=506),
    F_5: int = Form(..., ge=0, le=506),
    F_6: int = Form(..., ge=0, le=506),
    F_53: int = Form(..., ge=0, le=41),
    F_65: int = Form(..., ge=0, le=31),
    F_260: int = Form(..., ge=487, le=776),
    days_with_service: int = Form(..., ge=0, le=2164)
):
    # Perform predict
    data=CustomData(F_1,F_3,F_4,F_5,F_6,F_53,F_65,F_260,days_with_service)
    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    classification , proba = predict_pipeline.predict(pred_df)
    

    return templates.TemplateResponse("index.html",{"request": request, "result": classification, "proba" : proba})


if __name__ == "__main__":
    uvicorn.run(app)
