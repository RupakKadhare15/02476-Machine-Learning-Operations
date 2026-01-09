from toxic_comments.model import Model
from toxic_comments.data import MyDataset


def train():
    dataset = MyDataset('data/raw')
    model = Model()
    # add rest of your training code here


if __name__ == '__main__':
    train()
