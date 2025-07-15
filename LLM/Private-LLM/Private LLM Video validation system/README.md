## Project Structure:
video-validator/
├── app/
│   ├── main.py              # FastAPI app
│   ├── video_utils.py       # Video/audio processing
│   ├── whisper_transcriber.py
│   ├── frame_analyzer.py    # (YOLO/CLIP can go here)
│   └── llm_validator.py     # LLM integration with Ollama
├── models/                  # Optional directory for weights
├── Dockerfile
├── requirements.txt
└── README.md


## Requirements.txt
  - fastapi
  - uvicorn
  - openai-whisper
  - opencv-python
  - ffmpeg-python
  - requests
  - python-multipart



# DockerFile:
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git && \
    pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



## app/main.py
from fastapi import FastAPI, UploadFile, File
from app.video_utils import extract_audio, extract_frames
from app.whisper_transcriber import transcribe_audio
from app.llm_validator import validate_video

app = FastAPI()

@app.post("/validate/")
async def validate_video_endpoint(file: UploadFile = File(...)):
    video_path = f"temp/{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    # frames = extract_frames(video_path)  # optional vision step

    result = validate_video(transcript)
    return {"result": result}

## app/video_utils.py
import ffmpeg
import os

def extract_audio(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16k').run(overwrite_output=True)
    return audio_path

## app/whisper_transcriber.py
import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return result["text"]


## app/llm_validator.py
import requests

def validate_video(transcript: str) -> str:
    prompt = f"""
    You are a content moderator. Review this transcript for safety, violence, NSFW content, and age appropriateness:
    
    Transcript:
    {transcript}
    
    Return:
    - Safe or Unsafe
    - Reasoning
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "No response")


#  Run Ollama locally with ollama run mistral.

# Step 1: Start Ollama (if not already running)
ollama run mistral

# Step 2: Build and run the container
docker build -t video-validator .
docker run -p 8000:8000 video-validator

# Step 3: Upload a video for validation
curl -X POST "http://localhost:8000/validate/" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4"



# Running the whole system:
1. Start Ollama locally: Make sure it's listening at http://localhost:11434. ⚠️ Inside Docker, use host.docker.internal to access host Ollama from the container.
   - ollama pull mistral
   - ollama run mistral

2. Build and Run Docker
   - docker build -t video-validator
   - docker run -p 8000:8000 video-validator
  
3. Send a video for validation: use **curl** or **postman**.
   curl -X POST "http://localhost:8000/validate/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4"


## Next Steps (Optional Enhancements)
  - 🧠 Add YOLO or CLIP for visual validation
  - 🗃️ Add database (e.g., SQLite or Mongo) for audit logging
  - 🧪 Add unit tests with pytest
  - 🌐 Add frontend with Streamlit or React
  - 🔐 Add basic auth/token security for endpoints
