# main.py

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from gradio_client import Client

# Initialize FastAPI
app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load your Gradio client
gradio_client = Client("jarif/Bangla-English-Fake-News-Detection")

# Define request body structure for prediction
class TextRequest(BaseModel):
    text: str

# Define endpoint for prediction
@app.post("/predict")
async def predict_text(request: Request, text: str = Form(...)):
    try:
        # Make prediction using Gradio client
        prediction = gradio_client.predict(text=text, api_name="/predict")
        
        # Extract label and confidences from prediction
        label = prediction.get('label')
        confidences = prediction.get('confidences')

        # Render result.html with prediction result
        return templates.TemplateResponse("result.html", {"request": request, "label": label, "confidences": confidences})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define root endpoint to render HTML interface
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
