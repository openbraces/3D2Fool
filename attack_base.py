import os
import PIL
import PIL.Image
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from monodepth2 import networks  # Monodepth2 networks (encoder, decoder)
from monodepth2.utils import download_model_if_doesnt_exist
from data_loader_mde import (
    MyDataset,
)  # Custom dataset for rendering and texture application


# Wrapper for combining the encoder and decoder of Monodepth2
class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]  # Extract the disparity map at scale 0
        return disp


# Convert Monodepth2 disparity to actual depth (in meters)
def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


# Compute mean depth difference between adversarial and benign images in the masked region
def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask):
    scaler = 5.4  # Depth scaling factor
    dep1_adv = torch.clamp(
        disp_to_depth(torch.abs(adv_disp1), 0.1, 100)[1]
        * scene_car_mask.unsqueeze(0)
        * scaler,
        max=50,
    )
    dep2_ben = torch.clamp(
        disp_to_depth(torch.abs(ben_disp2), 0.1, 100)[1]
        * scene_car_mask.unsqueeze(0)
        * scaler,
        max=50,
    )
    mean_depth_diff = torch.sum(dep1_adv - dep2_ben) / torch.sum(scene_car_mask)
    return mean_depth_diff


# Compute the ratio of pixels affected by the attack (with error > 1m)
def get_affected_ratio(disp1, disp2, scene_car_mask):
    scaler = 5.4
    dep1 = torch.clamp(
        disp_to_depth(torch.abs(disp1), 0.1, 100)[1]
        * scene_car_mask.unsqueeze(0)
        * scaler,
        max=50,
    )
    dep2 = torch.clamp(
        disp_to_depth(torch.abs(disp2), 0.1, 100)[1]
        * scene_car_mask.unsqueeze(0)
        * scaler,
        max=50,
    )
    ones = torch.ones_like(dep1)
    zeros = torch.zeros_like(dep1)
    affected_ratio = torch.sum(
        scene_car_mask.unsqueeze(0) * torch.where((dep1 - dep2) > 1, ones, zeros)
    ) / torch.sum(scene_car_mask)
    return affected_ratio


# Total variation loss to encourage smoothness in the optimized texture
def loss_smooth(img):
    b, c, w, h = img.shape
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    return torch.square(torch.sum(s1 + s2)) / (b * c * w * h)


# Non-printability score loss: penalizes colors that are far from printable RGB colors
def loss_nps(img, color_set):
    _, h, w, c = img.shape
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)  # [B, 1, H, W, 3]
    color_set1 = color_set.unsqueeze(1).unsqueeze(1).unsqueeze(0)  # [1, C, 1, 1, 3]
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1) / 255, -1), 1).values
    return torch.sum(gap) / h / w


# Main adversarial texture training function
def attack(args):
    # Load Monodepth2 pretrained models
    model_name = "mono+stereo_1024x320"
    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4)
    )

    encoder.load_state_dict(
        {
            k: v
            for k, v in torch.load(encoder_path, map_location="cpu").items()
            if k in encoder.state_dict()
        }
    )
    depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location="cpu"))

    # Wrap and prepare the model
    depth_model = DepthModelWrapper(encoder, depth_decoder).to(args.device)
    depth_model.eval()
    for para in depth_model.parameters():
        para.requires_grad_(False)

    # Resize transforms
    feed_height, feed_width = 320, 1024
    input_resize = transforms.Resize([feed_height, feed_width])

    # Texture setup
    H, W = args.camou_shape, args.camou_shape
    resolution = 8
    h, w = int(H / resolution), int(W / resolution)

    # Create upsample kernel to scale optimized texture
    expand_kernel = torch.nn.ConvTranspose2d(
        3, 3, resolution, stride=resolution, padding=0
    ).to(args.device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)

    # Restricted color set (NPS)
    color_set = (
        torch.tensor(
            [
                [0, 0, 0],
                [255, 255, 255],
                [0, 18, 79],
                [5, 80, 214],
                [71, 178, 243],
                [178, 159, 211],
                [77, 58, 0],
                [211, 191, 167],
                [247, 110, 26],
                [110, 76, 16],
            ]
        )
        .to(args.device)
        .float()
        / 255
    )

    # Load initial texture from file
    camou_para = (
        transforms.ToTensor()(
            PIL.Image.open("texture_seed.png").resize((w, h)).convert("RGB")
        )
        .permute(1, 2, 0)
        .unsqueeze(0)
        .to(args.device)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    optimizer = optim.Adam([camou_para], lr=args.lr)
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # Load dataset
    dataset = MyDataset(
        args.train_dir, args.img_size, args.obj_name, args.camou_mask, args.device
    )
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    dataset.set_textures(camou_para1)

    # Training loop
    for epoch in tqdm(range(10), desc="Epochs", leave=False):
        for i, (index, total_img, total_img0, mask, img, *_) in enumerate(
            tqdm(loader, leave=False)
        ):
            input_image = input_resize(total_img)
            input_image0 = input_resize(total_img0)
            outputs = depth_model(input_image)

            # Mask and compute adversarial loss
            mask = input_resize(mask)[:, 0, :, :]
            adv_loss = torch.sum(10 * torch.pow(outputs * mask, 2)) / torch.sum(mask)
            tv_loss = loss_smooth(camou_para) * 0.1
            nps_loss = loss_nps(camou_para, color_set) * 5
            loss = tv_loss + adv_loss + nps_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Clamp and refresh texture
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1
            )
            camou_para1 = torch.clamp(camou_para1, 0, 1)
            dataset.set_textures(camou_para1)

        # Save visual and numpy output
        camou_png = cv2.cvtColor(
            (camou_para1[0].detach().cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(args.log_dir + str(epoch) + "camou.png", camou_png)
        np.save(
            args.log_dir + str(epoch) + "camou.npy", camou_para.detach().cpu().numpy()
        )


# Parse arguments and launch training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camou_mask",
        type=str,
        default="./car/mask.jpg",
        help="camouflage texture mask",
    )
    parser.add_argument(
        "--camou_shape", type=int, default=1024, help="shape of camouflage texture"
    )
    parser.add_argument(
        "--obj_name",
        type=str,
        default="./car/lexus_hs.obj",
        help="target object .obj file",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda:0"),
        help="training device",
    )
    parser.add_argument(
        "--train_dir", type=str, default="./data/", help="training image directory"
    )
    parser.add_argument(
        "--img_size", type=tuple, default=(320, 1024), help="training image size"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
    parser.add_argument("--lr", type=int, default=0.01, help="learning rate")
    parser.add_argument(
        "--log_dir", type=str, default="./res/", help="log output directory"
    )
    args = parser.parse_args()

    attack(args)
