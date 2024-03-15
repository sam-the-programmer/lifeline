import fastapi

print("Importing modules.")
import pickle
import os
import numpy as np

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


app = fastapi.FastAPI()


print("Creating callback manager.")
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

print("Loading model.")
llm = LlamaCpp(
    model_path="/kaggle/working/phi-2.Q3_K_M.gguf",
    temperature=0.5,  # high = expressive, low = accurate
    max_tokens=2000,
    n_threads=os.cpu_count,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
)

print("Loading embeddings.")
embed = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading metadata.")
with open("./data/metadata.pkl", "rb") as file:
    meta = pickle.load(file)
    meta_emb = np.asarray([i[0] for i in meta.items()], dtype=np.float32)
    meta_name = list(meta.values())

print("Definining search algorithm.")


def nearest(s):
    q = np.array(embed.embed_query(s.lower()))
    d = np.sum(
        np.square(meta_emb - np.array(q)), axis=1
    ).argmin()  # euclidean distance, but sqrt not needed
    return meta_name[d]


def load_nearest(s):
    with open("/kaggle/input/medical-dataset/" + nearest(s)) as file:
        return file.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: fastapi.WebSocket):
    await websocket.accept()
    query = await websocket.receive_text()
    query = load_nearest(query) + f"\nQ: {query}\nA:"
    if len(query) > 512:
        query = query[-512:]

    async for i in llm.bind(stop="Q:").astream(query):
        await websocket.send_text(i)
    await websocket.close(reason="Complete")
