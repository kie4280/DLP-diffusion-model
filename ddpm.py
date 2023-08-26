from diffusers import DDPMScheduler, UNet2DConditionModel
from argparse import ArgumentParser, Namespace
from dataloader import Training_dataset, LabelTransformer, Testing_dataset
from evaluator import evaluation_model
import torch
from torch.utils.data import DataLoader
import tqdm

IMG_SIZE=  (240, 320)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--beta", default=1e-3, type=float)
    parser.add_argument("--iters", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--regularization", default=1e-6, type=int)
    parser.add_argument("--test", action="store_true")

    return parser.parse_args()


def train(
    args,
    model: UNet2DConditionModel,
    train: DataLoader,
    test: DataLoader,
    evaluator: evaluation_model,
):
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.regularization
    )
    for epoch in range(1, args.epoch+1):
        tq = tqdm.tqdm(train)
        for (img, label) in tq:
            tq.set_description(f"epoch {epoch}")


    
    pass


def main(args):
    net = UNet2DConditionModel().to(args.device)
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
    train(args, net, training, testing, eval_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
