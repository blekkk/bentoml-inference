import bentoml
from bentoml.io import Image
from bentoml.io import PandasDataFrame


class Yolov8Runnable(bentoml.Runnable):
    def __init__(self):
        from ultralytics import YOLO
        from dotenv import load_dotenv
        import urllib.request
        import os

        load_dotenv()

        urllib.request.urlretrieve(
            os.getenv("MODEL_URL"),
            "model.pt",
        )

        self.model = YOLO("./model.pt")

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        # Return predictions only
        results = self.model(input_imgs)
        return results.pandas().xyxy

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs):
        # Return images with boxes and labels
        return self.model(input_imgs).render()


yolo_v8_runner = bentoml.Runner(Yolov8Runnable, max_batch_size=30)

svc = bentoml.Service("yolo_v5_demo", runners=[yolo_v8_runner])


@svc.api(input=Image(), output=PandasDataFrame())
async def invocation(input_img):
    batch_ret = await yolo_v8_runner.inference.async_run([input_img])
    return batch_ret[0]


@svc.api(input=Image(), output=Image())
async def render(input_img):
    batch_ret = await yolo_v8_runner.render.async_run([input_img])
    return batch_ret[0]
