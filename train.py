# coding=utf-8
from __future__ import absolute_import, division, print_function
import time
import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta

import torch
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends import cudnn

from utils.log import Logger
from utils.utils import *
from utils.lr_scheduler import adjust_learning_rate
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment, get_loader_dist


def setup(args, logger):
    # Prepare model
    model = AnomalyTransformer(win_size=args.win_size, enc_in=args.input_c, c_out=args.output_c, e_layers=3)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    logger.info(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def test(args, model, logger):
    # Distributed training
    if args.local_rank != -1:
        # model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)

    model.load_state_dict(
            torch.load(
                os.path.join(str(args.output_dir), str(args.dataset) + '_checkpoint.pth')))

    model.eval()
    temperature = 50
    criterion = nn.MSELoss(reduction='none')
    attens_energy = []
    train_loader = get_loader_dist(args=args, mode='train')
    thre_loader = get_loader_dist(args=args, mode='thre')

    logger.info("***** Running testing *****")
    logger.info("  Num steps = %d", len(thre_loader))
    logger.info("  Batch size = %d", args.batch_size)

    for i, (input_data_series, input_data_freq, labels) in enumerate(train_loader):
        input_data_series = input_data_series.float().to(args.device)
        input_data_freq = input_data_freq.float().to(args.device)

        output, series, prior, _ = model(input_data_series, input_data_freq)
        loss = torch.mean(criterion(input_data_series, output), dim=-1)

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
    
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    
    attens_energy = []
    for i, (input_data_series, input_data_freq, labels) in enumerate(thre_loader):
        input_data_series = input_data_series.float().to(args.device)
        input_data_freq = input_data_freq.float().to(args.device)

        output, series, prior, _ = model(input_data_series, input_data_freq)
        loss = torch.mean(criterion(input_data_series, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - args.anormly_ratio)
    logger.info(f"Threshold :{thresh}")
    
    test_labels = []
    attens_energy = []
    for i, (input_data_series, input_data_freq, labels) in enumerate(thre_loader):
        input_data_series = input_data_series.float().to(args.device)
        input_data_freq = input_data_freq.float().to(args.device)

        output, series, prior, _ = model(input_data_series, input_data_freq)
        loss = torch.mean(criterion(input_data_series, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        test_labels.append(labels)
        
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels)

    pred = (test_energy > thresh).astype(int)
    gt = test_labels.astype(int)
    
    logger.info(f"pred:   {pred.shape}")
    logger.info(f"gt:     {gt.shape}")
    
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    
    logger.info(f"pred: {pred.shape}")
    logger.info(f"gt:   {gt.shape}")
    
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
    logger.info(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
    return accuracy, precision, recall, f_score


def valid(args, model, val_loader, criterion, logger):
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(val_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    loss_1 = []
    loss_2 = []
    for i, (input_data_series, input_data_freq, _) in enumerate(val_loader):
        input_data_series = input_data_series.float().to(args.device)
        input_data_freq = input_data_freq.float().to(args.device)

        with torch.no_grad():
            output, series, prior, _ = model(input_data_series, input_data_freq)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)).detach())) + torch.mean(
                my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)).detach(),
                    series[u])))
            prior_loss += (torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    args.win_size)),
                            series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(),
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    args.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = criterion(output, input_data_series)
            loss_1.append((rec_loss - args.k * series_loss).item())
            loss_2.append((rec_loss + args.k * prior_loss).item())
    return np.average(loss_1), np.average(loss_2)

def train(args, model, logger):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader = get_loader_dist(args=args, mode='train')
    test_loader = get_loader_dist(args=args, mode='test')

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate)
    criterion = nn.MSELoss()

    t_total = args.num_steps

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        # model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=args.dataset)
    losses = AverageMeter()
    time_now = time.time()
    
    for epoch in range(t_total):
        iter_count = 0

        epoch_time = time.time()
        model.train()
        for i, (input_data_series, input_data_freq, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            iter_count += 1
            input_data_series = input_data_series.float().to(args.device)
            input_data_freq = input_data_freq.float().to(args.device)

            output, series, prior, _ = model(input_data_series, input_data_freq)
            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        args.win_size)).detach(),
                                series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    args.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = criterion(output, input_data_series)

            loss1 = rec_loss - args.k * series_loss
            loss2 = rec_loss + args.k * prior_loss

            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((t_total - epoch) * len(train_loader) - i)
                logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.gradient_accumulation_steps > 1:
                loss1 = loss1 / args.gradient_accumulation_steps
                loss2 = loss2 / args.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss1, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
                with amp.scale_loss(loss2, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss1.backward(retain_graph=True)
                loss2.backward()
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss1.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()


        logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = losses.avg
        losses.reset()
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=train_loss, global_step=epoch)

        vali_loss1, vali_loss2 = valid(args, model, val_loader=test_loader, criterion=criterion, logger = logger)
        if args.local_rank in [-1, 0]:
            writer.add_scalar("val/loss1", scalar_value=vali_loss1, global_step=epoch)
            writer.add_scalar("val/loss2", scalar_value=vali_loss2, global_step=epoch)

        logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, len(train_loader), train_loss, vali_loss1))

        early_stopping(vali_loss1, vali_loss2, model, args.output_dir)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        adjust_learning_rate(optimizer, epoch + 1, args.learning_rate)

    if args.local_rank in [-1, 0]:
        writer.close()



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--output_dir", default="checkpoint", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--batch_size", default=512, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--num_steps", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    cudnn.benchmark = True
    logger = Logger().get_logger(args)
    # Setup logging
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     filename='result.log',
    #                     filemode='a',
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args, logger)

    # Training
    if args.mode == 'train':
        train(args, model, logger)
    else:
        test(args, model, logger)

if __name__ == "__main__":
    main()
