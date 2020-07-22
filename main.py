import argparse
import tensorflow as tf
from tensorflow import keras
from models.MLP import MLP
from models.TCN import TCN
from utils import generate_synthetic_data
from sklearn.model_selection import train_test_split
from constants import Constants
from tqdm import tqdm
import time



def main(args):

    features, targets = generate_synthetic_data(args.model_type, args.num_samples)

    # split train/test sets
    x_train, x_val, y_train, y_val = train_test_split(features, targets, test_size=0.2)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size_train)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size_eval)

    if args.model_type == 'MLP':
        model = MLP(num_inputs=Constants._MLP_NUM_FEATURES, num_layers=Constants._MLP_NUM_LAYERS,
                    num_dims=Constants._MLP_NUM_DIMS, num_outputs=Constants._NUM_TARGETS, dropout_rate=args.dropout)
    elif args.model_type == 'TCN':
        model = TCN(nb_filters=Constants._TCN_NUM_FILTERS, kernel_size=Constants._TCN_KERNEL_SIZE,
                    nb_stacks=Constants._TCN_NUM_STACK, dilations=Constants._TCN_DIALATIONS,
                    padding=Constants._TCN_PADDING, dropout_rate=args.lr)

    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    for epoch in range(args.max_epoch):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print('Epoch: {}, Step: {}/{}, Loss: {}'.format(epoch, step, int(x_train.shape[0] / args.batch_size_train),
                                                                loss))

        # Perform inference and measure the speed every epoch
        start_time = time.time()
        for _, (x, _) in enumerate(db_val):
            _ = model.predict(x)
        end_time = time.time()

        print("Inference speed: {} samples/s\n".format(x_val.shape[0] / (end_time - start_time)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='TCN', help='Model name either MLP or TCN.')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples to generate.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--max_epoch', type=int, default=200, help='Max training epoch number.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Training batch size.')
    parser.add_argument('--batch_size_eval', type=int, default=128, help='Evaluating batch size.')

    args = parser.parse_args()
    main(args)
