
from fastapi import FastAPI
app = FastAPI()
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form
import os
import cv2

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/home")
async def read_index():
    return FileResponse("index.html")





@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...),):
    """
    Upload a file and return the file path.
    """
    # Save the file to disk
    file_path = f"{file.filename}"
    base_path = os.getcwd()
    localised_path = f"{base_path}/{file_path}"
    # Load the pre-trained face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(localised_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Reference distance for calculating face width and height
    reference_distance = 100  # In millimeters (adjust according to your specific scenario)
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Calculate face width and height in real-world measurements
        face_width_mm = w * reference_distance / image.shape[1]
        face_height_mm = h * reference_distance / image.shape[0]
        # Print the face dimensions
        # Print the face dimensions
        print("Face Width (mm):", face_width_mm)
        print("Face Height (mm):", face_height_mm)
        
        # Draw a rectangle around the face region of interest
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the image with the detected faces
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    return {"item": True , "Confidence":2 }
    