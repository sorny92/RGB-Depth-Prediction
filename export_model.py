import tensorflow as tf
import argparse
from model import DeeplabV3Plus


def generate_saved_model(model_weights, export_path):
    model = DeeplabV3Plus(512)
    print("LOADING THE MODEL")
    model.load_weights(model_weights)
    print("EXPORTING THE MODEL")
    model.export(export_path)


def quantize_model(saved_model_path, dataset_generator):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_generator
    converter.convert()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Export model',
        description='Export a tf checkpoint to savedModel and/or quantize the model')
    parser.add_argument('model_weights')
    parser.add_argument('export_path')
    args = parser.parse_args()

    generate_saved_model(args.model_weights, args.export_path)
