import pickle
import os
import pdb
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()


with open("dt_clf.pkl", "rb") as file:  # Changed the model name
    model = pickle.load(file)

# Assuming main.py is in the same directory as the templates and static directories
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


pdb.set_trace()  # Set breakpoint here
@app.post("/predict/")
async def predict(request: Request,
                  BMI: float = Form(...),
                  Smoking: int = Form(...),
                  AlcoholDrinking: int = Form(...),
                  Stroke: int = Form(...),
                  PhysicalHealth: int = Form(...),
                  MentalHealth: int = Form(...),
                  DiffWalking: int = Form(...),
                  Sex: int = Form(...),
                  AgeCategory: int = Form(...),
                  Race: int = Form(...),
                  Diabetic: int = Form(...),
                  PhysicalActivity: int = Form(...),
                  GenHealth: int = Form(...),
                  SleepTime: float = Form(...),
                  Asthma: int = Form(...),
                  KidneyDisease: int = Form(...),
                  SkinCancer: int = Form(...)):
pdb.set_trace()  # Set breakpoint here

    features = [BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory,    Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer]
    prediction = model.predict([features])[0]

    disease_status = "You have cardiac risk factors. Please check up ECG, Eco 2D and ETT" if prediction == 1 else "Not signs found for Heart Disease"

    return templates.TemplateResponse("results.html", {"request": request, "prediction": disease_status}, 
                                      headers={"Content-Type": "text/html; charset=utf-8"})


# Mounting the static files directory
@app.get("/static/{filename}")
async def get_static_file(filename: str):
    return FileResponse(os.path.join(static_dir, filename), media_type="text/css")  # Added media_type
