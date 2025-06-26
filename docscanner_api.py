from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
import numpy as np
import cv2
from PIL import Image
import io

from inference import Net, reload_seg_model, reload_rec_model

app = FastAPI()

# 加载模型（只加载一次）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
reload_seg_model(net.msk, './model_pretrained/seg.pth', device)
reload_rec_model(net.bm, './model_pretrained/DocScanner-L.pth', device)
net.eval()

def rectify_image(image: Image.Image):
    im_ori = np.array(image)[:, :, :3] / 255.
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)
    with torch.no_grad():
        bm = net(im.to(device))
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
        out = torch.nn.functional.grid_sample(
            torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(),
            lbl, align_corners=True
        )
        out_img = (((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1]).astype(np.uint8)
    return out_img

@app.post("/rectify/")
async def rectify(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result_img = rectify_image(image)
    result_pil = Image.fromarray(result_img).convert("RGB")
    buf = io.BytesIO()
    result_pil.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png") 