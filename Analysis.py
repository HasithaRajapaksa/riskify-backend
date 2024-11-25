from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import pandas as pd
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the OpenAI API key from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Pydantic model for the request body of /chatgpt/ endpoint
class ChatGPTRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"  # Default model can be overridden
    max_tokens: int = 50          # Limit the response length
    temperature: float = 0.7      # Creativity level

@app.post("/chatgpt/")
async def chatgpt(request: ChatGPTRequest):
    """
    Endpoint to send a prompt to the ChatGPT API and get a response.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # OpenAI API request payload
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    # Send POST request to OpenAI API
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, json=payload, headers=headers)

    # Return the OpenAI API response
    return response.json()


def generate_mock_tasks():
    tasks = [
        { "Task ID": 1, "Task Name": "Test1", "Duration (Days)": 10, "Dependencies": None },
        { "Task ID": 2, "Task Name": "Test2", "Duration (Days)": 5, "Dependencies": "1" },
        { "Task ID": 3, "Task Name": "Test3", "Duration (Days)": 8, "Dependencies": "1" },
        { "Task ID": 4, "Task Name": "Test4", "Duration (Days)": 6, "Dependencies": None },
        { "Task ID": 5, "Task Name": "Test5", "Duration (Days)": 7, "Dependencies": "4" },
        { "Task ID": 6, "Task Name": "Test6", "Duration (Days)": 12, "Dependencies": None },
        { "Task ID": 7, "Task Name": "Test7", "Duration (Days)": 4, "Dependencies": "5,6" },
        { "Task ID": 8, "Task Name": "Test8", "Duration (Days)": 10, "Dependencies": None },
        { "Task ID": 9, "Task Name": "Test9", "Duration (Days)": 9, "Dependencies": "8" },
        { "Task ID": 10, "Task Name": "Test10", "Duration (Days)": 3, "Dependencies": "9" }
    ]
    return tasks


@app.post("/chatgpt/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to accept an Excel file, read its content, log the data,
    and simulate sending it to the ChatGPT API.
    """
    try:
        # Validate file extension
        if not file.filename.endswith((".xls", ".xlsx")):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel file.")

        # Read the Excel file
        file_content = await file.read()
        df = pd.read_excel(file_content)

        # Log file details
        logging.info(f"File received: {file.filename}")
        logging.info(f"File size: {len(file_content)} bytes")

        # Log a preview of the data
        logging.info("Data preview: \n%s", df.head())

        # Simulate ChatGPT API call with data
        logging.info("Simulating ChatGPT API call with data: %s", df.to_dict(orient="records")[:5])

        # Prepare a mock response for frontend frinedly
        mock_tasks = generate_mock_tasks()

        # Return a preview of the processed data
        return {"message": "File processed successfully", "data_preview": mock_tasks, "crititical_path":"1,2,5,7", "suggested_advise": "You can complete your project without a risk if you follow 1 - 2 - 5 - 7 path."}

    except Exception as e:
        logging.error("Error processing the uploaded file: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process the file.")


class TaskRiskData(BaseModel):
    taskId: str
    duration: str

class RiskDataRequest(BaseModel):
    riskData: List[Dict[str, Any]]
    taskRiskData: TaskRiskData

@app.post("/receive-riskdata/")
async def receive_riskdata(request: RiskDataRequest):
    """
    Endpoint to receive JSON data with a structure like { "riskData": [], "taskRiskData": { "taskId": "", "duration": "" } }
    """
    try:
        # Log the received data
        print("Received riskData:", request.riskData)
        print("Received taskRiskData:", request.taskRiskData)

        # Process the data as needed
        # For example, you can iterate over the riskData list and perform operations

        return {"message": "Data received successfully", "received_data": request.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process the data.")