import argparse

import cv2
import numpy as np
import torch

from .model import build_model


def get_model(weight):
    cfg = SwinFaceCfg()
    model = build_model(cfg)
    dict_checkpoint = torch.load(weight)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])
    return model


@torch.inference_mode()
def single_image_inference(model: torch.nn.Module, img_tensor: torch.Tensor)-> torch.Tensor:
    output = model(img_tensor)  # .numpy()
    return output['Recognition']

def image_transform(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img /= 255.0
    img -= 0.5
    img /= 0.5
    return img

@torch.inference_mode()
def inference(weight, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()
    img_tensor.div_(255).sub_(0.5).div_(0.5)

    model = get_model(weight)
    output = model(img_tensor)  # .numpy()

    for each in output.keys():
        print(each, "\t", output[each][0].numpy())


class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for j in range(11)]
    fam_feature = "all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str,
                        default='<your path>/checkpoint_step_79999_gpu_0.pt')
    parser.add_argument('--img', type=str, default="<your path>/test.jpg")
    args = parser.parse_args()
    inference(args.weight, args.img)
