import flask

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


app = flask.Flask(__name__)
model = None


def get_custom_backbone_fast_rcnn(num_classes):
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=4)
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def load_model(model_weights):
    global model
    model = get_custom_backbone_fast_rcnn(4)
    model.load_state_dict(torch.load(
        model_weights, map_location=torch.device('cpu')))
    model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"]
            img = Image.open(img)
            img = np.array(img)
            img = torch.tensor(img).permute(
                2, 0, 1).unsqueeze(0).float().to(device)
            outputs = model(img)

            # Apply NMS here
            keeps = torchvision.ops.nms(
                outputs[0]['boxes'], outputs[0]['scores'], 0.1).to('cpu').numpy()

            boxes = [list(x.detach().to('cpu').numpy().astype(str))
                     for idx, x in enumerate(outputs[0]['boxes']) if idx in keeps]
            labels = [str(int(x)) for idx, x in enumerate(
                outputs[0]['labels']) if idx in keeps]
            scores = [str(float(x)) for idx, x in enumerate(
                outputs[0]['scores']) if idx in keeps]

            annotations = {"boxes": boxes, "labels": labels, "scores": scores}
            predictions = flask.jsonify(annotations)

            data["success"] = True

        if flask.request.files.getlist('images[]'):
            processed_imgs = []
            imgs = flask.request.files.getlist('images[]')
            for img in imgs:
                img = Image.open(img)
                img = np.array(img)
                img = torch.tensor(img).permute(2, 0, 1).float()
                processed_imgs.append(img)

            outputs = model(processed_imgs)

            predictions = {}
            cameras = ['FRONT_LEFT', 'FRONT',
                       'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
            for idx, camera in enumerate(cameras):
                keeps = torchvision.ops.nms(
                    outputs[idx]['boxes'], outputs[idx]['scores'], 0.1).to('cpu').numpy()

                boxes = [list(x.detach().numpy().astype(str)) for idx, x in enumerate(
                    outputs[idx]['boxes']) if idx in keeps]
                labels = [str(int(x)) for idx, x in enumerate(
                    outputs[idx]['labels']) if idx in keeps]
                scores = [str(float(x)) for idx, x in enumerate(
                    outputs[idx]['scores']) if idx in keeps]

                annotations = {"boxes": boxes,
                               "labels": labels, "scores": scores}
                predictions[camera] = annotations

            predictions = flask.jsonify(predictions)
            data["success"] = True

    return predictions


if __name__ == "__main__":
    print("* Loading trained model and Flask server...")
    load_model('/app/model_weights.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    app.run(host='0.0.0.0', port=5000, debug=False)
