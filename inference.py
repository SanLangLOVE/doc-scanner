from model import DocScanner
from seg import U2NETP

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import argparse
import time

import warnings
warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.msk = U2NETP(3, 1)
        self.bm = DocScanner()  # 矫正

    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.bm(x, iters=12, test_mode=True)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm


def reload_seg_model(model, path="", device=None):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def reload_rec_model(model, path="", device=None):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(seg_model_path, rec_model_path, distorrted_path, save_path):
    # distorted images list
    img_list = [f for f in os.listdir(distorrted_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

    # creat save path for rectified images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net init
    net = Net().to(device)
    # reload seg model
    reload_seg_model(net.msk, seg_model_path, device)
    # reload rec model
    reload_rec_model(net.bm, rec_model_path, device)

    total_time = 0
    count = 0
    profiled = False  # 只 profile 一次

    # 新增：用 schedule + tensorboard_trace_handler 生成 event 文件
    from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
    profile_steps = 4  # 只 profile 前4张图片
    if len(img_list) >= profile_steps:
        net.eval()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=tensorboard_trace_handler('./tb_log'),
            record_shapes=True
        ) as prof:
            for step, img_path in enumerate(img_list[:profile_steps]):
                name = os.path.splitext(img_path)[0]
                img_path_full = os.path.join(distorrted_path, img_path)
                try:
                    im_ori = np.array(Image.open(img_path_full))[:, :, :3] / 255.
                except Exception as e:
                    print(f"[Warning] Failed to open {img_path_full}: {e}")
                    continue
                h, w, _ = im_ori.shape
                im = cv2.resize(im_ori, (288, 288))
                im = im.transpose(2, 0, 1)
                im = torch.from_numpy(im).float().unsqueeze(0)
                start_time = time.time()
                with record_function("model_inference"):
                    with torch.no_grad():
                        bm = net(im.to(device))
                        bm = bm.cpu()
                        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
                        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
                        bm0 = cv2.blur(bm0, (3, 3))
                        bm1 = cv2.blur(bm1, (3, 3))
                        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
                        out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
                        cv2.imwrite(os.path.join(save_path, name + '_rec.png'), (((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1]).astype(np.uint8))
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"[Profile] {img_path} | 分辨率: {w}x{h} | 推理耗时: {elapsed:.4f} 秒")
                total_time += elapsed
                count += 1
                prof.step()
        print("已生成 tb_log 目录（含 event 文件），可用 TensorBoard 自动显示 Profile。")
        # 剩余图片正常推理
        img_list_remain = img_list[profile_steps:]
    else:
        img_list_remain = img_list

    # 其余图片正常推理
    for img_path in img_list_remain:
        name = os.path.splitext(img_path)[0]
        img_path_full = os.path.join(distorrted_path, img_path)
        try:
            im_ori = np.array(Image.open(img_path_full))[:, :, :3] / 255.
        except Exception as e:
            print(f"[Warning] Failed to open {img_path_full}: {e}")
            continue
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        start_time = time.time()
        with torch.no_grad():
            bm = net(im.to(device))
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
            cv2.imwrite(os.path.join(save_path, name + '_rec.png'), (((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1]).astype(np.uint8))
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{img_path} | 分辨率: {w}x{h} | 推理耗时: {elapsed:.4f} 秒")
        total_time += elapsed
        count += 1

    if count > 0:
        print(f"平均每张图片推理耗时: {total_time/count:.4f} 秒")
        print(f"总耗时: {total_time:.4f} 秒, 总图片数: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_model_path', default='./model_pretrained/seg.pth')
    parser.add_argument('--rec_model_path', default='./model_pretrained/DocScanner-L.pth')
    parser.add_argument('--distorrted_path', default='./distorted/')
    parser.add_argument('--rectified_path', default='./rectified/')
    opt = parser.parse_args()

    rec(seg_model_path=opt.seg_model_path,
        rec_model_path=opt.rec_model_path,
        distorrted_path=opt.distorrted_path,
        save_path=opt.rectified_path)


if __name__ == "__main__":
    main()
