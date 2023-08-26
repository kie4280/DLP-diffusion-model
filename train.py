from diffusers import DDPMScheduler, UNet2DModel
from argparse import ArgumentParser, Namespace
from .dataloader import Training_dataset, LabelTransformer, Testing_dataset
from.evaluator import evaluation_model


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()

def train(args, model, train, test, evaluator):
    pass

def main(args):
    net = UNet2DModel().to()
    training = Training_dataset
    testing = Testing_dataset("test.json")
    eval_model  = evaluation_model()
    train(args, net, training, testing, eval_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
