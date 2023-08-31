from diffusers import DDPMScheduler, UNet2DModel
from argparse import ArgumentParser, Namespace
from dataloader import Training_dataset, LabelTransformer, Testing_dataset
from evaluator import evaluation_model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import os
from torchvision import transforms
import torchvision
import numpy as np

IMG_SIZE = (240, 320)
TQDM_COL = 120


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--beta", default=1e-3, type=float)
    parser.add_argument("--train_iters", default=1000, type=int)
    parser.add_argument("--infer_iters", default=500, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--regularization", default=0, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--save_img", default=True, action="store_true")

    return parser.parse_args()


def train(
    args,
    model: UNet2DModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    scheduler: DDPMScheduler,
    # evaluator: evaluation_model,
):
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.regularization
    )

    for epoch in range(1, args.epoch + 1):
        tq = tqdm.tqdm(train_dataloader, ncols=TQDM_COL)
        for img, label in tq:
            img = img.to(args.device)
            label = label.to(args.device)
            epsilon = torch.randn(img.shape).to(args.device)
            timesteps = torch.randint(0, args.train_iters, size=(label.shape[0],)).to(
                dtype=torch.int, device=args.device
            )
            noisy_image = scheduler.add_noise(img, epsilon, timesteps)

            optim.zero_grad()
            pred_noise = model(noisy_image, timesteps, label).sample
            loss = torch.nn.functional.mse_loss(pred_noise, epsilon)
            loss.backward()
            optim.step()
            tq.set_description(f"epoch {epoch}")
            tq.set_postfix({"loss": loss.detach().cpu().item()})

        # acc = test(args, model, test_dataloader, scheduler, evaluator)
        torch.save(model.state_dict(), f"checkpoints/{epoch}.pth")
        if args.save_img and epoch % 3 == 0:
            generate_img(args, model, test_dataloader, scheduler)


def generate_img(
    args,
    model: UNet2DModel,
    test_dataloader: DataLoader,
    scheduler: DDPMScheduler,
):
    tq = tqdm.tqdm(test_dataloader, ncols=TQDM_COL)
    images = []
    step_size = np.floor(args.infer_iters / 11)
    targets = np.arange(
        start=0, stop=step_size * 11, step=step_size
    )
    for batch_idx, label in enumerate(tq):
        label = label.to(args.device)
        scheduler.set_timesteps(args.infer_iters)
        noisy_img = torch.randn(size=(label.shape[0], 3, 64, 64)).to(args.device)
        progressive = torch.zeros((11, 3, 64, 64)).to(args.device)
        progress_idx = 0

        for idx, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                pred_noise = model(noisy_img, t, label).sample

            s = scheduler.step(pred_noise, t, noisy_img)
            noisy_img = s.prev_sample
            if idx <= targets[-1] and idx == targets[progress_idx]:
                progressive[progress_idx] = s.pred_original_sample[0]
                progress_idx += 1
        images.append(noisy_img)

    progressive = torchvision.utils.make_grid(progressive, nrow=11)
    torchvision.utils.save_image(progressive, f"images/generation.png")
    grid = torchvision.utils.make_grid(torch.cat(images, dim=0))
    torchvision.utils.save_image(grid, f"images/visual.png")


def test(
    args,
    model: UNet2DModel,
    test_dataloader: DataLoader,
    scheduler: DDPMScheduler,
    evaluator: evaluation_model,
):
    tq = tqdm.tqdm(test_dataloader, ncols=TQDM_COL)
    total = 0
    transf = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for i, label in enumerate(tq):
        label = label.to(args.device)
        scheduler.set_timesteps(1000)
        noisy_img = torch.randn(size=(label.shape[0], 3, 64, 64)).to(args.device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                pred_noise = model(noisy_img, t, label).sample
            noisy_img = scheduler.step(pred_noise, t, noisy_img).prev_sample

        total += evaluator.eval(transf(noisy_img), label)

    return total / len(test_dataloader)


def main(args):
    os.makedirs("checkpoints/", exist_ok=True)
    os.makedirs("images/", exist_ok=True)
    net = UNet2DModel(
        sample_size=(64, 64),
        in_channels=3,
        out_channels=3,
        class_embed_type=None,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(args.device)
    del net.class_embedding
    net.class_embedding = nn.Linear(24, 128 * 4).to(args.device)

    training = DataLoader(
        Training_dataset(), batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    testing = DataLoader(
        Testing_dataset("new_test.json"),
        batch_size=8,
        shuffle=False,
        num_workers=4,
    )
    scheduler = DDPMScheduler(
        args.train_iters,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
    )

    if args.ckpt is not None:
        l = torch.load(args.ckpt)
        net.load_state_dict(l)

    if args.test:
        eval_model = evaluation_model()
        generate_img(args, net, testing, scheduler)
        avg_acc = test(args, net, testing, scheduler, eval_model)
        print(f"average score {avg_acc}")
        return
    train(args, net, training, testing, scheduler)


if __name__ == "__main__":
    args = parse_args()
    main(args)
