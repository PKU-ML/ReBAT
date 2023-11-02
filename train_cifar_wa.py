import argparse
import logging
import sys
import time
import math
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter
from attacks import AttackerPolymer
from utils import *

from networks.vgg import VGG16
from networks.mobilenetv2 import MobileNetV2 as MobileV2
from networks.wideresnet import WideResNet
from networks.preactresnet import PreActResNet18

parser = argparse.ArgumentParser()

# general
parser.add_argument('--fname', default='cifar', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--val', action='store_true')  # use validation-based early stopping

# evaluation
parser.add_argument('--eval', action='store_true')  # evaluation mode
parser.add_argument('--eval-last-only', action='store_true')
parser.add_argument('--eval-best-only', action='store_true')
parser.add_argument('--eval-online', action='store_true')
parser.add_argument('--eval-train-robust', action='store_true')

# model
parser.add_argument('--model', default='PreActResNet18', choices=['PreActResNet18', 'WideResNet', 'VGG16', 'MobileNet'])
parser.add_argument('--width-factor', default=10, type=int)  # for WRN
parser.add_argument('--resume', default=0, type=int)  # resume from this epoch
parser.add_argument('--load-folder', default=None,
                    type=str)  # can specify a folder to load checkpoints; if not specified, load from default folder
parser.add_argument('--save-path', type=str, default='exps')
parser.add_argument('--chkpt-iters', default=10, type=int)  # checkpoint save frequency

# WA model
parser.add_argument('--decay-rate', default=0.999, type=float)
parser.add_argument('--warmup-epochs', default=105, type=int)

# dataset
parser.add_argument('--data-dir', default='../cifar-data', type=str)
parser.add_argument('--num-classes', default=10, type=int)  # set to 100 for CIFAR 100

# data augmentation (CutMix)
parser.add_argument('--cutmix', action='store_true')
parser.add_argument('--cutmix-alpha', type=float, default=1.0)
parser.add_argument('--cutmix-beta', type=float, default=1.0)

# training
parser.add_argument('--l2', default=0, type=float)
parser.add_argument('--l1', default=0, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)

# learning rate
parser.add_argument('--lr-schedule', default='piecewise', choices=['piecewise', 'linear', 'cosine', 'constant'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-factor', type=float, default=1.5)  # decay factor for piecewise schedule
parser.add_argument('--stage1', type=int, default=100)
parser.add_argument('--stage2', type=int, default=150)

# attacker
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
parser.add_argument('--eval-attack', default='pgd', type=str, choices=['pgd', 'none'])
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--attack-iters', default=10, type=int)
parser.add_argument('--restarts', default=1, type=int)
parser.add_argument('--pgd-alpha', default=2, type=float)
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])

# stronger attacker for ReBAT++
parser.add_argument('--stronger-attack', action='store_true')
parser.add_argument('--stronger-epsilon', default=10, type=int)
parser.add_argument('--stronger-attack-iters', default=12, type=int)
parser.add_argument('--stronger-eval', action='store_true')  # also use stronger attack during evaluation

# BoAT regularization
parser.add_argument('--use-reg-schedule',
                    action='store_true')  # if set to False, by default it stays constant as args.beta
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--beta-factor', type=float, default=1.5)  # multiply factor in piecewise schedule
parser.add_argument('--reg-schedule', default='dependent', choices=['piecewise', 'dependent'])

args = parser.parse_args()


def main():
    # ------------------ basic settings ------------------
    args.fname = args.save_path + '/' + args.fname  # collect all the experiments
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.val:  # please use --val for validation-based early stopping
        try:
            dataset = torch.load("cifar10_validation_split.pth") if args.num_classes == 10 else torch.load(
                "cifar100_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data'] / 255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar(args.data_dir, num_classes=args.num_classes)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4) / 255.),
                         dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data'] / 255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    # ------------------ attacker ------------------
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    Attackers = AttackerPolymer(epsilon, args.attack_iters, pgd_alpha, args.num_classes, device)

    # ------------------ model ------------------
    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=args.num_classes)
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    elif args.model == 'VGG16':
        model = VGG16(n_classes=args.num_classes)
    elif args.model == 'MobileNet':
        model = MobileV2(num_classes=args.num_classes)
    else:
        raise ValueError("Unknown model")

    model.train()
    model.to(device)

    from copy import deepcopy
    model_wa = deepcopy(model)

    # ------------------ optimizer ------------------
    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.l2},
                  {'params': no_decay, 'weight_decay': 0}]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # ------------------ learning rate decay schedule ------------------
    if args.lr_schedule == 'constant':
        lr_schedule = lambda t: args.lr_max
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t < args.stage1:
                return args.lr_max
            elif t < args.stage2:
                return args.lr_max / args.lr_factor
            else:
                return args.lr_max / args.lr_factor ** 2
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                          [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    else:
        raise NotImplementedError("Unknown LR decay schedule!")

    # ------------------ BoAT regularization strength schedule ------------------
    # will only be used when args.use_reg_schedule=True; by default it stays constant as args.beta
    if args.reg_schedule == 'piecewise':
        def reg_schedule(t):
            if t < args.stage2:  # WA and BoAT regularization start after the first LR decay, usually at epoch 105
                return args.beta
            else:
                return args.beta * args.beta_factor
    elif args.reg_schedule == 'dependent':
        def reg_schedule(t):
            rate = lr_schedule(t)
            return (args.lr_max / rate - 1) / 2
    else:
        raise NotImplementedError("Unknown regularization schedule!")

    # ------------------ preparation for training ------------------
    best_test_robust_acc = 0
    best_test_robust_acc_wa = 0
    start_epoch = 0
    epochs = args.epochs

    # resume from checkpoints
    if args.resume:
        if args.load_folder is None:
            args.load_folder = args.fname
        start_epoch = args.resume
        logger.info(f'Resuming at epoch {start_epoch}')

        # load optimizer and online model weights
        model.load_state_dict(torch.load(os.path.join(args.load_folder, f'model_{start_epoch - 1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.load_folder, f'opt_{start_epoch - 1}.pth')))

        # load WA model weights
        try:  # after args.warmup_epochs, the WA model is different from the online model, so WA model weights and online model weights are saved into two different files
            model_wa.load_state_dict(torch.load(os.path.join(args.load_folder, f'wa_model_{start_epoch - 1}.pth')))
        except:  # before args.warmup_epochs, the WA model is the same as the online model, so we only save one copy of the weights and load the online model weights for the WA model
            model_wa.load_state_dict(torch.load(os.path.join(args.load_folder, f'model_{start_epoch - 1}.pth')))

    # ------------------ start training ------------------
    if not args.eval:
        log_dir = os.path.join(args.fname, 'tblog', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)  # Tensorboard
        logger.info(
            'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

    for epoch in range(start_epoch, epochs):
        if args.eval:  # in evaluation mode, just skip the training loop
            break
        model.train()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_reg_loss = 0
        train_robust_acc = 0
        train_n = 0
        decay_rate = args.decay_rate if epoch >= args.warmup_epochs else 0.  # for WA
        beta = args.beta if epoch >= args.warmup_epochs else 0.  # force deactivating BoAT regularization before WA starts
        start_time = time.time()
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.cutmix:
                X, y_a, y_b, lam = cutmix_data(X, y, args.cutmix_alpha, args.cutmix_beta)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))

            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                if not args.stronger_attack or epoch < args.stage1:  # ReBAT[strong]
                    if args.cutmix:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts,
                                           args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts,
                                           args.norm)
                else:  # ReBAT (without stronger attack)
                    if args.cutmix:
                        delta = attack_pgd(model, X, y, args.stronger_epsilon / 255, pgd_alpha,
                                           args.stronger_attack_iters, args.restarts,
                                           args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(model, X, y, args.stronger_epsilon / 255, pgd_alpha,
                                           args.stronger_attack_iters, args.restarts,
                                           args.norm)
                delta = delta.detach()
            elif args.attack == 'none':  # standard training
                delta = torch.zeros_like(X)

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            if args.cutmix:
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = criterion(robust_output, y)

            reg_loss = torch.tensor(0.).cuda()
            if beta > 0:  # apply BoAT loss
                with torch.no_grad():
                    robust_output_wa = model_wa(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                reg_loss = F.kl_div(F.log_softmax(robust_output, dim=1),
                                    F.softmax(robust_output_wa, dim=1),
                                    reduction='batchmean')
                if reg_loss < 1e10:
                    if args.use_reg_schedule:
                        beta = reg_schedule(epoch + (i + 1) / len(train_batches))
                    robust_loss += reg_loss * beta

            if args.l1:
                for name, param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1 * param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            output = model(normalize(X))
            if args.cutmix:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            moving_average(model_wa, model, decay_rate, update_bn=True)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_reg_loss += reg_loss.item() * y.size(0)
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        # evaluate one model, can be the online/WA model
        def val(model, prefix=''):
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            true_y, pred_y, pred_y_rob = [], [], []
            batches = val_batches if args.val else test_batches

            for i, batch in enumerate(batches):
                X, y = batch['input'], batch['target']

                if args.eval_attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    if args.stronger_attack and args.stronger_eval:
                        delta = attack_pgd(model, X, y, args.stronger_epsilon / 255, pgd_alpha,
                                           args.stronger_attack_iters, args.restarts,
                                           args.norm, early_stop=args.eval)
                    else:
                        delta = attack_pgd(model, X, y, 8.0 / 255.0, 2.0 / 255, 10, args.restarts, args.norm,
                                           early_stop=args.eval)
                delta = delta.detach()

                output = model(normalize(X))
                loss = criterion(output, y)

                if args.eval_attack == 'none':
                    robust_output = output
                    robust_loss = loss
                else:
                    robust_output = model(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    robust_loss = criterion(robust_output, y)

                true_y.append(y.cpu())
                pred_y.append(output.argmax(1).cpu())
                pred_y_rob.append(robust_output.argmax(1).cpu())

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            test_time = time.time()

            writer.add_scalar(f'{prefix}val/acc', test_acc / test_n, epoch)
            writer.add_scalar(f'{prefix}val/loss', test_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}val/robust_loss', test_robust_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}val/robust_acc', test_robust_acc / test_n, epoch)
            writer.add_scalar(f'{prefix}train/acc', train_acc / train_n, epoch)
            logger.info(
                '[%s] %d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                prefix, epoch, train_time - start_time, test_time - train_time, lr,
                               train_loss / train_n, train_acc / train_n, train_robust_loss / train_n,
                               train_robust_acc / train_n,
                               test_loss / test_n, test_acc / test_n, test_robust_loss / test_n,
                               test_robust_acc / test_n)

            # save checkpoint
            if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'{prefix}model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'{prefix}opt_{epoch}.pth'))

            return test_acc / test_n, test_loss / test_n, test_robust_acc / test_n, test_robust_loss / test_n

        # evaluate on training data (for WA model)
        def val_train(model, prefix=''):
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            true_y, pred_y, pred_y_rob = [], [], []

            for i, batch in enumerate(train_batches):
                X, y = batch['input'], batch['target']

                if args.eval_attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, 8.0 / 255.0, 2.0 / 255, 10, args.restarts, args.norm,
                                       early_stop=args.eval)
                delta = delta.detach()

                output = model(normalize(X))
                loss = criterion(output, y)

                if args.eval_attack == 'none':
                    robust_output = output
                    robust_loss = loss
                else:
                    robust_output = model(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    robust_loss = criterion(robust_output, y)

                true_y.append(y.cpu())
                pred_y.append(output.argmax(1).cpu())
                pred_y_rob.append(robust_output.argmax(1).cpu())

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            test_time = time.time()

            true_y = torch.cat(true_y)
            pred_y = torch.cat(pred_y)
            pred_y_rob = torch.cat(pred_y_rob)
            try:
                record_stats(true_y, pred_y, pred_y_rob, writer, epoch, prefix=f'{prefix}val')
            except:
                pass

            writer.add_scalar(f'{prefix}train/acc', test_acc / test_n, epoch)
            writer.add_scalar(f'{prefix}train/loss', test_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}train/robust_loss', test_robust_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}train/robust_acc', test_robust_acc / test_n, epoch)
            logger.info(
                '[%s] %d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                prefix, epoch, train_time - start_time, test_time - train_time, lr,
                               train_loss / train_n, train_acc / train_n, train_robust_loss / train_n,
                               train_robust_acc / train_n,
                               test_loss / test_n, test_acc / test_n, test_robust_loss / test_n,
                               test_robust_acc / test_n)

            # save checkpoint
            if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'{prefix}model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'{prefix}opt_{epoch}.pth'))

            return test_acc / test_n, test_loss / test_n, test_robust_acc / test_n, test_robust_loss / test_n

        writer.add_scalar('train/loss', train_loss / train_n, epoch)
        writer.add_scalar('train/robust_loss', train_robust_loss / train_n, epoch)
        writer.add_scalar('train/reg_loss', train_reg_loss / train_n, epoch)
        writer.add_scalar('train/robust_acc', train_robust_acc / train_n, epoch)

        test_acc, test_loss, test_robust_acc, test_robust_loss = val(model)
        if epoch >= args.warmup_epochs:
            test_acc_wa, test_loss_wa, test_robust_acc_wa, test_robust_loss_wa = val(model_wa, prefix='wa_')
            if args.eval_train_robust:
                val_train(model_wa, prefix='wa_')
        else:
            test_acc_wa, test_loss_wa, test_robust_acc_wa, test_robust_loss_wa = test_acc, test_loss, test_robust_acc, test_robust_loss

        # save best
        if test_robust_acc > best_test_robust_acc:
            print(f"update best online model! Current online best: {best_test_robust_acc} -> {test_robust_acc}")
            torch.save(model.state_dict(), os.path.join(args.fname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc
        # save best
        if test_robust_acc_wa > best_test_robust_acc_wa:
            print(f"update best WA model! Current WA best: {best_test_robust_acc_wa} -> {test_robust_acc_wa}")
            torch.save(model_wa.state_dict(), os.path.join(args.fname, f'wa_model_best.pth'))
            best_test_robust_acc_wa = test_robust_acc_wa

    # ------------------ final evaluation ------------------
    print("Evaluating best and last...")
    logger.info(' \t '.join(['Mode', 'clean', 'PGD20', 'AA']))

    # last
    if not args.eval_best_only:
        print("Now evaluating last...")
        if args.eval_online:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'model_{args.epochs - 1}.pth')))
        else:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'wa_model_{args.epochs - 1}.pth')))
        res_list = attack_all(model_wa, test_batches, Attackers)
        logger.info('%s \t ' + '%.4f \t ' * 2 + '%.4f', '[last wa]', *res_list)

    # best
    if not args.eval_last_only:
        print("Now evaluating best...")
        if args.eval_online:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'model_best.pth')))
        else:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'wa_model_best.pth')))
        res_list = attack_all(model_wa, test_batches, Attackers)
        logger.info('%s \t ' + '%.4f \t ' * 2 + '%.4f', '[best wa]', *res_list)


# ------------------ evaluation functions ------------------
# borrowed from SEAT - https://arxiv.org/abs/2203.09678
class AverageMeter(object):
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


class NormInputModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(normalize(X))


def attack_all(model, test_loader, Attackers):
    model = NormInputModel(model)
    model.eval()

    clean_accuracy = AverageMeter()
    pgd20_accuracy = AverageMeter()
    # pgd100_accuracy = AverageMeter()
    # mim_accuracy = AverageMeter()
    # cw_accuracy = AverageMeter()
    # APGD_ce_accuracy = AverageMeter()
    # APGD_dlr_accuracy = AverageMeter()
    # APGD_t_accuracy = AverageMeter()
    # FAB_t_accuracy = AverageMeter()
    # Square_accuracy = AverageMeter()
    aa_accuracy = AverageMeter()
    from tqdm import tqdm
    from collections import OrderedDict
    pbar = tqdm(test_loader)
    pbar.set_description('Attacking all')

    for batch_idx, batch in enumerate(pbar):
        pbar_dic = OrderedDict()
        inputs, targets = batch['input'], batch['target']

        acc_dict = Attackers.run_all(model, inputs, targets)

        clean_accuracy.update(acc_dict['NAT'][0].item(), inputs.size(0))
        pgd20_accuracy.update(acc_dict['PGD_20'][0].item(), inputs.size(0))
        # pgd100_accuracy.update(acc_dict['PGD_100'][0].item(), inputs.size(0))
        # mim_accuracy.update(acc_dict['MIM'][0].item(), inputs.size(0))
        # cw_accuracy.update(acc_dict['CW'][0].item(), inputs.size(0))
        # APGD_ce_accuracy.update(acc_dict['APGD_ce'][0].item(), inputs.size(0))
        # APGD_dlr_accuracy.update(acc_dict['APGD_dlr'][0].item(), inputs.size(0))
        # APGD_t_accuracy.update(acc_dict['APGD_t'][0].item(), inputs.size(0))
        # FAB_t_accuracy.update(acc_dict['FAB_t'][0].item(), inputs.size(0))
        # Square_accuracy.update(acc_dict['Square'][0].item(), inputs.size(0))
        aa_accuracy.update(acc_dict['AA'][0].item(), inputs.size(0))

        pbar_dic['clean'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['PGD20'] = '{:.2f}'.format(pgd20_accuracy.mean)
        # pbar_dic['PGD100'] = '{:.2f}'.format(pgd100_accuracy.mean)
        # pbar_dic['MIM'] = '{:.2f}'.format(mim_accuracy.mean)
        # pbar_dic['CW'] = '{:.2f}'.format(cw_accuracy.mean)
        # pbar_dic['APGD_ce'] = '{:.2f}'.format(APGD_ce_accuracy.mean)
        # pbar_dic['APGD_dlr'] = '{:.2f}'.format(APGD_dlr_accuracy.mean)
        # pbar_dic['APGD_t'] = '{:.2f}'.format(APGD_t_accuracy.mean)
        # pbar_dic['FAB_t'] = '{:.2f}'.format(FAB_t_accuracy.mean)
        # pbar_dic['Square'] = '{:.2f}'.format(Square_accuracy.mean)
        pbar_dic['AA'] = '{:.2f}'.format(aa_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    # return [clean_accuracy.mean, pgd20_accuracy.mean, pgd100_accuracy.mean, mim_accuracy.mean, cw_accuracy.mean,
    #         APGD_ce_accuracy.mean, APGD_dlr_accuracy.mean, APGD_t_accuracy.mean, FAB_t_accuracy.mean,
    #         Square_accuracy.mean, aa_accuracy.mean]
    return [clean_accuracy.mean, pgd20_accuracy.mean, aa_accuracy.mean]


if __name__ == "__main__":
    main()
