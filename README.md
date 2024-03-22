# Lifeline

## Setup
To run it:
- Install the required python dependencies with...
  - `pip install uvicorn[standard] fastapi langchain llama-cpp-python numpy`
- Download the *phi-2.Q2_K.gguf* file from `https://huggingface.co/TheBloke/phi-2-GGUF/tree/main`, or the *ggml-model-Q4_0.gguf* file from `https://huggingface.co/BioMistral/BioMistral-7B-GGUF/tree/main` in the directory. If you use the BioMistral model, change the server code so that model is loaded.

## Usage
- Run the server with the command line command `python -m uvicorn demo:app`
- Open the `app.html` file in a browser.
- Use the UI