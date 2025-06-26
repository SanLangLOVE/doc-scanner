import streamlit as st
import os
from PIL import Image
import torch
import numpy as np
import cv2

from inference import Net, reload_seg_model, reload_rec_model

# 初始化模型（只加载一次）
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    reload_seg_model(net.msk, './model_pretrained/seg.pth', device)
    reload_rec_model(net.bm, './model_pretrained/DocScanner-L.pth', device)
    net.eval()
    return net, device

def rectify_image(net, device, image: Image.Image):
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

# Streamlit 页面
st.title("SOYUAN-PicScanner-文档图片自动矫正+彩色扫描算法在线验证")
st.write("上传一张拍摄的文档图片，自动拉矫正生成彩色扫描件")

uploaded_file = st.file_uploader("选择一张图片", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", use_column_width=True)
    st.write("正在处理，请稍候...")

    net, device = load_model()
    result_img = rectify_image(net, device, image)
    st.image(result_img, caption="s扫描矫正后图片", use_column_width=True)
    st.success("处理完成！")
    # 下载按钮
    result_pil = Image.fromarray(result_img).convert("RGB")
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format='PNG')
    st.download_button(
        label="下载扫描矫正图片",
        data=buf.getvalue(),
        file_name="rectified.png",
        mime="image/png"
    ) 