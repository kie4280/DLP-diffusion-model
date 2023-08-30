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

IMG_SIZE = (240, 320)
TQDM_COL = 120


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--beta", default=1e-3, type=float)
    parser.add_argument("--train_iters", default=1000, type=int)
    parser.add_argument("--infer_iters", default=30, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--regularization", default=1e-6, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ckpt", default=None)

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


def test(
    args,
    model: UNet2DModel,
    test_dataloader: DataLoader,
    scheduler: DDPMScheduler,
    evaluator: evaluation_model,
):
    tq = tqdm.tqdm(test_dataloader, ncols=TQDM_COL)
    total = 0
    transf = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    for i, label in enumerate(tq):
        label = label.to(args.device)
        scheduler.set_timesteps(args.infer_iters)
        noisy_img = torch.randn(size=(label.shape[0], 3, *IMG_SIZE)).to(args.device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                pred_noise = model(noisy_img, t, label).sample
            noisy_img = scheduler.step(pred_noise, t, noisy_img).prev_sample

        total += evaluator.eval(transf(noisy_img), label)
        grid = torchvision.utils.make_grid(noisy_img)
        torchvision.utils.save_image(grid, f"images/visual_{i}.png")

    return total / len(test_dataloader.dataset)


def main(args):
    os.makedirs("checkpoints/", exist_ok=True)
    os.makedirs("images/", exist_ok=True)
    net = UNet2DModel(
        sample_size=IMG_SIZE,
        in_channels=3,
        out_channels=3,
        class_embed_type=None,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            32,
            64,
            128,
            256,
            512,
        ),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
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
        ),
    ).to(args.device)
    net.class_embedding = nn.Linear(24, 32 * 4).to(args.device)

    training = DataLoader(
        Training_dataset(), batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    testing = DataLoader(
        Testing_dataset("test.json"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    eval_model = evaluation_model()
    scheduler = DDPMScheduler(args.train_iters)

    if args.ckpt is not None:
        l = torch.load(args.ckpt)
        net.load_state_dict(l)

    if args.test:
        test(args, net, testing, scheduler, eval_model)
        return
    train(args, net, training, testing, scheduler)


if __name__ == "__main__":
    args = parse_args()
    main(args)
