from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import pandas as pd
import logging
from dotenv import load_dotenv

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
    max_tokens: int = 500         # Adjusted for longer responses
    temperature: float = 0.7


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


@app.post("/chatgpt/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to accept an Excel file, read its content, log the data,
    format it as a prompt, and send it to the ChatGPT API for critical path analysis.
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

        # Log the extracted data
        logging.info("Extracted data:\n%s", df)

        # Ensure required columns exist in the Excel data
        required_columns = {"Task ID", "Task Name", "Duration(Days)", "Dependencies"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Ensure the Excel file contains {required_columns}."
            )

        # Generate the dynamic table format for the prompt
        prompt_table = "\n".join(
            f"{row['Task ID']} -> {row['Task Name']} -> {row['Duration(Days)']} -> {row['Dependencies']}"
            for _, row in df.iterrows()
        )

        # Log the dynamically created table
        logging.info("Formatted table for prompt:\n%s", prompt_table)

        # Construct the ChatGPT prompt
        prompt = (
            "Find the list of activities in the critical path and the list of activities in the least critical path. "
            "Ensure to provide the final answer in below format.\n"
            "Critical Path Activities: Activities as comma separated values\n"
            "Least critical path activities: Activities as comma separated values\n"
            "Details will be provided in the following order: Task ID, Task Name, Duration, Dependencies. "
            "If there are no dependencies, this will be indicated with the word ‘No’.\n"
            f"{prompt_table}\n"
            "Note: Consider the least critical path as the path whose total float addition is the highest."
        )

        # Log the final prompt
        logging.info("Final ChatGPT prompt:\n%s", prompt)

        # Prepare the ChatGPT API request payload
        chatgpt_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        # Define headers for ChatGPT API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        # Send the request to ChatGPT API
        async with httpx.AsyncClient() as client:
            chatgpt_response = await client.post(OPENAI_API_URL, json=chatgpt_payload, headers=headers)

        # Check if ChatGPT API call was successful
        if chatgpt_response.status_code != 200:
            logging.error("ChatGPT API Error: %s", chatgpt_response.text)
            raise HTTPException(status_code=chatgpt_response.status_code, detail="Error from ChatGPT API")

        # Parse ChatGPT response
        chatgpt_result = chatgpt_response.json()

        # Extract and log the ChatGPT result
        critical_path_analysis = chatgpt_result["choices"][0]["message"]["content"]
        logging.info("Critical Path Response from ChatGPT:\n%s", critical_path_analysis)

        # Return the original Excel data and critical path analysis
        return {
            "message": "File processed successfully and sent to ChatGPT",
            "critical_path_analysis": critical_path_analysis,
        }

    except Exception as e:
        logging.error("Error processing the uploaded file: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process the file.")



@app.post("/chatgpt/process")
async def process_and_send_to_chatgpt(file: UploadFile = File(...)):
    """
    Endpoint to process the uploaded Excel file, generate a prompt,
    send it to ChatGPT API, and return the response.
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

        # Log the extracted data
        logging.info("Extracted data:\n%s", df)

        # Ensure required columns exist in the Excel data
        required_columns = {"Task ID", "Task Name", "Duration(Days)", "Dependencies"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Ensure the Excel file contains {required_columns}."
            )

        # Generate the dynamic table format for the prompt
        prompt_table = "\n".join(
            f"{row['Task ID']} -> {row['Task Name']} -> {row['Duration(Days)']} -> {row['Dependencies']}"
            for _, row in df.iterrows()
        )

        # Log the dynamically created table
        logging.info("Formatted table for prompt:\n%s", prompt_table)

        # Construct the ChatGPT prompt
        prompt = (
            "Find the list of activities in the critical path and the list of activities in the least critical path. "
            "Ensure to provide the final answer in below format.\n"
            "Critical Path Activities: Activities as comma separated values\n"
            "Least critical path activities: Activities as comma separated values\n"
            "Details will be provided in the following order: Task ID, Task Name, Duration, Dependencies. "
            "If there are no dependencies, this will be indicated with the word ‘No’.\n"
            f"{prompt_table}\n"
            "Note: Consider the least critical path as the path whose total float addition is the highest."
            "Ensure that we get in the following format without any other details and not required a additional description.\n"
            "Critical Path Activities: Activities as comma separated values\n"
            "Least critical path activities: Activities as comma separated values\n"
            "Critical path is the longest path of activities whose float is 0."
        )

        # prompt = """ Find the list of activities in the critical path and the list of activities in the least critical path. ( By path ID) Ensure to provide the final answer in below format. o Critical Path Activities: Activities as comma separated values o Least critical path activities: Activities as comma separated values Details will be provided in the following order: Task ID, Task Name, Duration, Dependencies. If there are no dependencies, this will be indicated with the word ‘No’. o A -> Task 1 -> 5 -> No o B -> Task 2 -> 8 -> A o C -> Task 3 -> 6 -> A o D -> Task 4 -> 8 -> C o E -> Task 5 -> 12 -> B o F -> Task 6 -> 7 -> B o G -> Task 7 -> 10 -> D,F o H -> Task 8 -> 8 -> E,G Note: Consider the least critical path as the path whose total float addition is the highest. only provide the final answer with critical path and least critical path activities as comma separated values. """


        # Log the final prompt
        logging.info("Final ChatGPT prompt:\n%s", prompt)

        # Prepare the ChatGPT API request payload
        payload = {
            "model": "gpt-4o-2024-05-13",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        # "max_tokens": 300,

        # Send the prompt to ChatGPT API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(OPENAI_API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            logging.error("Error from ChatGPT API: %s", response.text)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to process prompt. Error: {response.text}"
            )

        # Extract and return the response from ChatGPT
        chatgpt_response = response.json()
        logging.info("ChatGPT API response:\n%s", chatgpt_response)

        # Extract only the 'content' from the ChatGPT response
        if "choices" in chatgpt_response and len(chatgpt_response["choices"]) > 0:
            content = chatgpt_response["choices"][0]["message"]["content"]
        else:
            logging.error("Invalid response structure from ChatGPT API.")
            raise HTTPException(
                status_code=500,
                detail="Invalid response structure from ChatGPT API."
            )

        # Return the response content
        return {
            "message": "Prompt processed successfully.",
            "data": content,
        }

    except Exception as e:
        logging.error("Error processing the uploaded file: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process the file.")