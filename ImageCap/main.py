import argparse

from ImageCap.train import train
from ImageCap.eval import evaluate_coco_captioning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning Model")
    
    subparsers = parser.add_subparsers(dest="mode", help="Mode")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate_coco_captioning()
    else:
        parser.print_help() 