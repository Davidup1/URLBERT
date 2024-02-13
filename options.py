import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size of data")
    parser.add_argument('--lr', type=float, default=0.00001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight_decay in training")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    return args