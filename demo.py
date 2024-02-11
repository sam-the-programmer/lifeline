import asyncio
import fastapi

app = fastapi.FastAPI()

example = """Hello, I am a large language model chatbot.\nNext line...""".split(" ")

@app.websocket("/ws")
async def websocket_endpoint(websocket: fastapi.WebSocket):
    await websocket.accept()
    query = await websocket.receive_text()
    print(query)
    for i in example:
        t = i+" "
        t = t.replace("\n", "<br>")

        await websocket.send_text(t)
        await asyncio.sleep(0.1)
    await websocket.close(reason="Complete")