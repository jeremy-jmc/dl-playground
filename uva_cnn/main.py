import argparse
import lightning as L
L.seed_everything(42)


parser = argparse.ArgumentParser(description='Train models')
parser.add_argument('--model', type=str, choices=['googlenet', 'resnet', 'resnetpreact', 'densenet'], help='Model to train', required=True)
parser.add_argument('--max-epochs', type=int, default=20, help='Maximum number of epochs')
args = parser.parse_args()

from train import *

if args.model == 'googlenet':
    googlenet_model, googlenet_results = train_model(
        model_name="GoogleNet",
        model_hparams={"num_classes": 10, "act_fn_name": "relu"},
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        max_epochs=args.max_epochs,
    )
    print("GoogleNet Results", googlenet_results)
elif args.model == 'resnet':
    resnet_model, resnet_results = train_model(
        model_name="ResNet",
        model_hparams={"num_classes": 10, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        max_epochs=args.max_epochs,
    )
    print("ResNet Results", resnet_results)
elif args.model == 'resnetpreact':
    resnetpreact_model, resnetpreact_results = train_model(
        model_name="ResNet",
        model_hparams={
            "num_classes": 10,
            "c_hidden": [16, 32, 64],
            "num_blocks": [3, 3, 3],
            "act_fn_name": "relu",
            "block_name": "PreActResNetBlock",
        },
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        save_name="ResNetPreAct",
        max_epochs=args.max_epochs,
    )
    print("ResNetPreAct Results", resnetpreact_results)
elif args.model == 'densenet':
    densenet_model, densenet_results = train_model(
        model_name="DenseNet",
        model_hparams={
            "num_classes": 10,
            "num_layers": [6, 6, 6, 6],
            "bn_size": 2,
            "growth_rate": 16,
            "act_fn_name": "relu",
        },
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        max_epochs=args.max_epochs,
    )
torch.cuda.empty_cache()


"""
https://github.com/vllm-project/vllm/issues/1726
"""