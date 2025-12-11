from config import *
from modelling import predict_test, train_model
from resnet_model import BetterBirdCNN

if __name__ == "__main__":
    train_model(BetterBirdCNN, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    predict_test(BetterBirdCNN, batch_size=BATCH_SIZE)