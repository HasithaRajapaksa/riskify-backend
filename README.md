
# FastAPI Project Setup and Run Guide

This document provides the steps to set up and run the FastAPI project locally.

---

## Steps to Run the Project Locally

1. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the FastAPI application**:
    ```sh
    uvicorn Analysis:app --reload --port 5000
    ```

5. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000`.

---

Ensure you have the `.env` file with the necessary environment variables, such as `OPENAI_API_KEY`, in the project root directory.

