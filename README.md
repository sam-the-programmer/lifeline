# Lifeline

## Setup
To run it:
- Install the required python dependencies with...
  - `pip install uvicorn[standard]`
  - `pip install fastapi`
- Install the **Live Server** VS Code extension and click the little _Go Live_ button in the bottom status bar.
- Open the terminal and run `uvicorn main:app --reload`
- Open another terminal and run `npx tailwindcss -i ./web/input.css -o ./web/output.css --watch`
- Go to [the URL of the server + "app.html"](https://127.0.0.1:5000/app.html) (you can click on the link on the left).
