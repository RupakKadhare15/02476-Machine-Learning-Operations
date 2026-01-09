from toxic_comments.model import Model
from toxic_comments.data import MyDataset

def train():
    _ = MyDataset("data/raw") # _ => dataset
    _ = Model() # _ => model
    # add rest of your training code here
    pass

if __name__ == "__main__":
    train()
