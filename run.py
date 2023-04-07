"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from optimizers.regularizers import all_regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10","Wiki16k"],
    help="Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="RescalH", choices=all_models, help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=all_regularizers, default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--gcn_dropout", default=0, type=float, help="GCN_Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--distance", default="dist", type=str, choices=["dist", "dot", "none"], help="distance function"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--curvature", action='store_true', help="Whether to learn the curvature"
)
parser.add_argument(
    "--xvaier", action='store_true', help="Whether to learn the curvature"
)
parser.add_argument(
    "--no_act", action='store_true', help="Whether have the activation function"
)
parser.add_argument(
    "--no_gcn", action='store_true', help="Whether have the activation function"
)
parser.add_argument(
    "--double_layer", action='store_true', help="Whether have the double layer"
)
parser.add_argument(
    "--self_loop", action='store_true', help="Whether have the self loop"
)
parser.add_argument(
    "--weight", action='store_true', help="ce_weight"
)
parser.add_argument(
    "--sparse", action='store_true', help="ce_weight"
)
parser.add_argument(
    "--num_head", default=1, type=int, help="gcn attention head num"
)
parser.add_argument(
    "--smoothing", default=0.0, type=float, help="label smoothing"
)
parser.add_argument(
    "--init_c", default=1.0, type=float, help="label smoothing"
)
parser.add_argument(
    "--reg_w", default=0, type=float, help="GCN aggregation mode"
)

def train(args):

    device = "cuda"
    save_dir = get_savedir(args.model, args.dataset)
    
    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    
    fhlr = logging.FileHandler(os.path.join(save_dir, "train.log"))
    fhlr.setLevel(logging.INFO)
    fhlr.setFormatter(formatter)
    logging.getLogger("").addHandler(fhlr)
    logging.info("Saving logs in: {}".format(save_dir))
    
    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()
    if args.weight:
        ce_weight = torch.Tensor(dataset.get_weight()).cuda()
        if args.dtype == 'double':
            ce_weight = ce_weight.double()
    else:
        ce_weight = None
        
    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()
    edge_index, edge_type = dataset.construct_adj(device)
    
    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)
    
    logging.info(f'json_conf: {json.dumps(vars(args))}')
    # create model
    model = getattr(models, args.model)(args, edge_index, edge_type)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    
    model.to(device)

    # get optimizer
    if args.regularizer != "DURA_RESCAL_P" and args.regularizer != "DURA_DistMultH_P":
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
    else:
        regularizer = getattr(regularizers, args.regularizer)(args.reg, args.reg_w)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg), smoothing = args.smoothing, weight=ce_weight)
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")

    train_examples_test = torch.from_numpy(dataset.data["train"].astype("int64"))

    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:

            train_metrics = avg_both(*model.compute_metrics(train_examples_test, filters))
            logging.info(format_metrics(train_metrics, split="train"))
            
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))
            
            
            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.cuda()
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())
