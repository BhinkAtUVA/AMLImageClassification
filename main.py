from config import *
from modelling import predict_test, train_model
from resnet_model import BirdResNet34

if __name__ == "__main__":
    train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, dropout_p=0.2)
    predict_test(batch_size=BATCH_SIZE)
