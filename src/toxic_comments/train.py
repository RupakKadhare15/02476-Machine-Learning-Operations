from toxic_comments.data import MyDataset
from toxic_comments.model import Model


def train():
    _ = MyDataset('data/raw')
    _ = Model()
    # add rest of your training code here


if __name__ == '__main__':
    train()
