import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = (
    "https://drive.google.com/uc?export=download&id=140cBy4v2pekBIudZEMUryTE08_PYmb2P"
)
export_file_name = "filipino_food_model_resnet34.pkl"

classes = [
    "lumpia",
    "sinigang",
    "afritada",
    "cassava_cake",
    "pancit_palabok",
    "ube_milkshake",
    "pork_adobo",
    "chicken_bistek",
    "bistek_talagog",
    "chicharon",
    "bibingka",
    "pork_sisig",
    "kare-kare",
    "halo-halo",
    "lechon",
    "kaldereta",
    "arroz_caldo",
    "longganisa",
    "tocino",
    "leche_flan",
    "filipino_spaghetti",
    "chicken_adobo",
    "crispy_pata",
    "chicken_inasal",
    "bulalo",
    "tinola",
    "tapa",    del learn
    "laing",
    "pinakbet",
    "bagnet",
    "bicol_express",
    "balut",
    "kinilaw",
    "liempo",
    "champorado",
    "buco_pie",
    "turon",
    "pandesal",
    "taho",
    "pancit_guisado",
    "pork_barbecue",
    "ginataang_gulay",
    "lechon_kawali",
]
path = Path(__file__).parent

app = Starlette()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["X-Requested-With", "Content-Type"],
)
app.mount("/static", StaticFiles(directory="app/static"))


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, "wb") as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and "CPU-only machine" in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route("/")
async def homepage(request):
    html_file = path / "view" / "index.html"
    return HTMLResponse(html_file.open().read())


@app.route("/analyze", methods=["POST"])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data["file"].read())
    img = open_image(BytesIO(img_bytes))
    result = learn.predict(img)
    prediction = result[0]
    confidence = max(result[2])
    return JSONResponse({"result": str(prediction), "confidence": str(confidence)})


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app=app, host="0.0.0.0", port=5000, log_level="info")
