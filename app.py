from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory=".")
model = YOLO('yolov8n.pt')  # Load the YOLOv8 model

# Directory to save uploaded and processed images
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

# Create directories if they do not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)  # Perform inference
        annotated_frame = results[0].plot()  # Annotate the frame with detections
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/upload_image')
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded image
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process the image with YOLO model
    img = cv2.imread(file_location)
    results = model(img)
    annotated_img = results[0].plot()

    # Save the processed image
    processed_image_path = os.path.join(PROCESSED_DIR, file.filename)
    cv2.imwrite(processed_image_path, annotated_img)

    # Return the path of the processed image
    return {"output_image": f"/processed/{file.filename}"}

@app.get('/processed/{filename}')
async def get_processed_image(filename: str):
    return StreamingResponse(open(os.path.join(PROCESSED_DIR, filename), "rb"), media_type="image/jpeg")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
