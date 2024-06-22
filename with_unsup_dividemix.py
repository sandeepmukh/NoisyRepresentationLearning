from __future__ import print_function
import os

default_n_threads = 8
os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"

import sys
import copy
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Sampler, WeightedRandomSampler
import torchvision.models as models
import random
import math
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from datetime import datetime
import os
import wandb

import torch.distributed as dist 
import torch.multiprocessing as mp
#from torch.nn.parallel import DistrbutedDataParallel as DDP

import utilities.dataloader_inat as dataloader
from utilities.general_utils import LDAMLoss, get_class_counts
from utilities.cmo_utils import cut_mix
from utilities.gnn_utils import (
    train_gnn,
    eval_train_gnn,
    test_with_gnn,
    val_gnn,
)

def get_arg_parser():
    parser = argparse.ArgumentParser(description="iNat", add_help=False)
    parser.add_argument("--batch_size", default=128, type=int,
                        help="train batchsize")
    parser.add_argument(
        "--lr", "--learning_rate", default=0.0001, type=float,
        help="initial learning rate"
    )
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="parameter for Beta")
    parser.add_argument(
        "--lambda_u", default=0, type=float, help="deprecated"
    )
    parser.add_argument(
        "--p_threshold", default=0.5, type=float,
        help="clean probability threshold"
    )
    parser.add_argument(
        "--class_cond_epoch", default=20, type=int,
        help='Number of epochs to use class conditional splitting')
    parser.add_argument("--T", default=0.5, type=float,
                        help="sharpening temperature")
    parser.add_argument("--num_epochs", default=80, type=int)
    parser.add_argument("--id", default="train_inat")
    parser.add_argument(
        "--data_path", default="./train_inat", type=str, help="path to dataset"
    )
    parser.add_argument("--seed", default=123)
    parser.add_argument("--gpuid", default=5, type=int)
    parser.add_argument("--num_class", default=1000, type=int)
    parser.add_argument("--num_batches", default=100, type=int)
    parser.add_argument("--restart_epoch", default=0, type=int)
    parser.add_argument("--restart_out_dir", default=None, type=str)
    parser.add_argument("--warmup", default=1, type=int)
    parser.add_argument("--ldam", action="store_true", default=False)
    parser.add_argument("--out_dir", default="checkpoint/inat", type=str)
    parser.add_argument("--weighted_alpha", default=1.0, type=float)
    parser.add_argument("--cmo_prob", default=0.5, type=float)
    parser.add_argument("--use_cmo", action="store_true", default=False)
    parser.add_argument("--cmo_beta", default=1.0, type=float)
    parser.add_argument("--gnn_epochs", default=1, type=int)
    parser.add_argument("--use_gnn", action="store_true", default=False)
    parser.add_argument("--use_prop_split", action="store_true", default=False)
    parser.add_argument("--use_mixup", action="store_true", default=True)
    parser.add_argument("--use_neg_entropy_penalty", 
                        action="store_true", default=False)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    return parser

# FROM JigsawViT CODE


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
                        backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

####################


# Training
def train(
    epoch,
    net,
    net2,
    optimizer,
    labeled_trainloader,
    unlabeled_trainloader,
    CEloss,
    conf_penalty,
    cmo_loader=None,
):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    use_cmo = args.use_cmo and cmo_loader is not None
    if use_cmo:
        cmo_iter = iter(cmo_loader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(
        labeled_trainloader
    ):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.__next__()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.__next__()

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1
        )

        if use_cmo:
            rand = np.random.rand()
            if rand < args.cmo_prob:
                try:
                    cmo_inputs, _, cmo_labels, _ = cmo_iter.__next__()
                except:
                    cmo_iter = iter(cmo_loader)
                    cmo_inputs, _, cmo_labels, _ = cmo_iter.__next__()
                cmo_labels = torch.zeros(batch_size, args.num_class).scatter_(
                    1, cmo_labels.view(-1, 1), 1
                )
                inputs_x2, targets_x2 = cut_mix(
                    inputs_x2, labels_x, cmo_inputs, cmo_labels, args
                )
            else:
                targets_x2 = labels_x

            inputs_x2, targets_x2 = inputs_x2.cuda(), targets_x2.cuda()

        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = (
            inputs_x.cuda(),
            inputs_x2.cuda(),
            labels_x.cuda(),
            w_x.cuda(),
        )
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (
                torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2,
                                                                dim=1)
            ) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        # mixmatch
        if args.use_mixup:
            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2],
                                   dim=0)
            if use_cmo:
                all_targets = torch.cat(
                    [targets_x, targets_x2, targets_u, targets_u], dim=0
                )
            else:
                all_targets = torch.cat(
                    [targets_x, targets_x, targets_u, targets_u], dim=0
                )
        else:
            all_inputs = torch.cat([inputs_u, inputs_u2])
            all_targets = torch.cat([targets_u, targets_u])
            supervised_loss = CEloss(
                torch.cat([outputs_x, outputs_x2]),
                torch.cat([labels_x, labels_x])
            )

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = (
            l * input_a[: batch_size * 2] + (1 - l) *
            input_b[: batch_size * 2]
        )
        mixed_target = (
            l * target_a[: batch_size * 2] + (1 - l) *
            target_b[: batch_size * 2]
        )

        logits = net(mixed_input)

        Lx = -torch.mean(torch.sum(
                        F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # regularization
        if args.use_neg_entropy_penalty:
            penalty = conf_penalty(logits)
        else:
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + penalty
        if not args.use_mixup:
            loss = loss + supervised_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write("\r")
        sys.stdout.write(
            "Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f "
            % (epoch, args.num_epochs, batch_idx + 1, num_iter, Lx.item())
        )
        sys.stdout.flush()


def warmup(net, optimizer, dataloader, epoch, CEloss, conf_penalty):
    net.train()
    if epoch == 0:
        for param in net.parameters():
            param.requires_grad = False

        #for param in net.module.classifier.parameters():
        #for param in net.head.parameters():
        for param in net.module.head.parameters():
            param.requires_grad = True
            param.requires_grad = True
    else:
        for param in net.parameters():
            param.requires_grad = True

    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        if batch_idx > 200:
            for param in net.parameters():
                param.requires_grad = True

        penalty = conf_penalty(outputs)
        L = loss + penalty
        L.backward()
        optimizer.step()

        sys.stdout.write("\r")
        sys.stdout.write(
            "|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f"
            % (batch_idx + 1, args.num_batches, loss.item(), penalty.item())
        )
        sys.stdout.flush()


def val(net, val_loader, test_loader, k, best_acc):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    val_acc = 100.0 * correct / total
    print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, val_acc))
    print(f"\n| Best Accuracy for Net{k}: {best_acc[k - 1]:.2f}%")
    wandb.log({"val_acc": val_acc})
    if val_acc > best_acc[k - 1]:
        best_acc[k - 1] = val_acc
        print("| Saving Best Net%d ..." % k)
        save_point = f"{args.out_dir}/%s_net%d.pth.tar" % (args.id, k)
        torch.save(net.state_dict(), save_point)
        
        
    #test version
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            test_total += targets.size(0)
            test_correct += predicted.eq(targets).cpu().sum().item()
    test_acc = 100.0 * test_correct / test_total
    print("\n| Test\t Net%d  Acc: %.2f%%" % (k, test_acc))
    wandb.log({"test_acc": test_acc})
    if test_acc > best_acc[k - 1]:
        best_acc[k - 1] = test_acc
        print("| Saving Best Net%d ..." % k)
        save_point = f"{args.out_dir}/%s_net%d.pth.tar" % (args.id, k)
        torch.save(net.state_dict(), save_point)
    
    return val_acc


def test(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.0 * correct / total
    print("\n| Test Acc: %.2f%%\n" % (acc))
    return acc


# def eval_train(epoch, model, eval_loader, CE):
#     model.eval()
#     num_samples = len(eval_loader.dataset)
#     local_losses = []
#     local_ys = []
#     local_paths = []
    
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, paths) in enumerate(eval_loader):
#             inputs, targets = inputs.cuda(), targets.cuda()
#             outputs = model(inputs)
#             loss = CE(outputs, targets)
#             local_losses.extend(loss.cpu().numpy())
#             local_ys.extend(targets.cpu().numpy())
#             local_paths.extend(paths)
            
#             sys.stdout.write("\r")
#             sys.stdout.write("| Evaluating loss Iter %3d\t" % (batch_idx))
#             sys.stdout.flush()

#     # Converting lists to numpy arrays for consistency
#     local_losses = np.array(local_losses)
#     local_ys = np.array(local_ys)

#     # Gathering all data from all GPUs
#     global_losses = gather_all_data(local_losses)
#     global_ys = gather_all_data(local_ys)
#     global_paths = gather_all_data(local_paths)

#     # Debugging output to verify lengths
#     print(f"Length of global_losses: {len(global_losses)}")
#     print(f"Length of global_ys: {len(global_ys)}")
#     print(f"Length of global_paths: {len(global_paths)}")
    
#     assert len(global_losses) == len(global_ys) == len(global_paths), "Lengths do not match"
    
#     prob = np.zeros(len(global_ys))
#     if epoch < args.class_cond_epoch:
#         for y in set(global_ys):
#             idx = global_ys == y
#             if idx.sum() == 1:
#                 prob[idx] = 0
#                 continue
#             curr_losses = (global_losses[idx] - global_losses[idx].min()) / (
#                 global_losses[idx].max() - global_losses[idx].min() + 1e-8
#             )
#             curr_losses = curr_losses.reshape(-1, 1)
#             gmm = GaussianMixture(
#                         n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
#             gmm.fit(curr_losses)
#             curr_prob = gmm.predict_proba(curr_losses)
#             prob[idx] = curr_prob[:, gmm.means_.argmin()]
#     else:
#         global_losses = (global_losses - global_losses.min()) / (global_losses.max() - global_losses.min())
#         global_losses = global_losses.reshape(-1, 1)
#         gmm = GaussianMixture(
#                         n_components=2, max_iter=15, reg_covar=5e-4, tol=1e-2)
#         gmm.fit(global_losses)
#         prob = gmm.predict_proba(global_losses)
#         prob = prob[:, gmm.means_.argmin()]

#     return prob, global_paths

def eval_train(epoch, model, eval_loader, CE, args):
    model.eval()
    local_losses, local_ys, local_paths = [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets, paths) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            
            local_losses.extend(loss.cpu().numpy())
            local_ys.extend(targets.cpu().numpy())
            local_paths.extend(paths)
            
            sys.stdout.write(f"\r| Evaluating loss Iter {batch_idx:3d}\t")
            sys.stdout.flush()

    # Convert lists to tensors for efficient gathering
    local_losses_tensor = torch.tensor(local_losses).cuda()
    local_ys_tensor = torch.tensor(local_ys).cuda()
    dist.barrier()
    
    global_losses = gather_all_data_tensor(local_losses_tensor).cpu().numpy()
    global_ys = gather_all_data_tensor(local_ys_tensor).cpu().numpy()
    global_paths = gather_all_data_strings(local_paths)  # Gathering all string data from all GPUs

    # Debugging output to verify lengths
    print(f"Length of global_losses: {len(global_losses)}")
    print(f"Length of global_ys: {len(global_ys)}")
    print(f"Length of global_paths: {len(global_paths)}")
    
    assert len(global_losses) == len(global_ys) == len(global_paths), "Lengths do not match"
    
    prob = np.zeros(len(global_ys))
    if epoch < args.class_cond_epoch:
        for y in set(global_ys):
            idx = global_ys == y
            print(f"Class {y}: {idx.sum()} samples")  # Debugging statement
            if idx.sum() < 2:  # Check if there are fewer than 2 samples
                prob[idx] = 0
                continue

            curr_losses = global_losses[idx]
            curr_losses = (curr_losses - curr_losses.min()) / (curr_losses.max() - curr_losses.min() + 1e-8)
            curr_losses = curr_losses.reshape(-1, 1)
            
            gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
            gmm.fit(curr_losses)
            curr_prob = gmm.predict_proba(curr_losses)
            prob[idx] = curr_prob[:, gmm.means_.argmin()]
    else:
        if len(global_losses) >= 2:  # Check if there are at least 2 samples
            global_losses = (global_losses - global_losses.min()) / (
                global_losses.max() - global_losses.min()
            ).reshape(-1, 1)
            
            gmm = GaussianMixture(n_components=2, max_iter=15, reg_covar=5e-4, tol=1e-2)
            gmm.fit(global_losses)
            prob = gmm.predict_proba(global_losses)[:, gmm.means_.argmin()]
        else:
            prob = np.zeros(len(global_losses))  # Default to zero probabilities if fewer than 2 samples

    return prob, global_paths



# def gather_all_data(local_data):
#     """
#     Gathers all data from all GPUs.
#     """
#     global_data = [None for _ in range(dist.get_world_size())]
#     dist.all_gather_object(global_data, local_data)
#     global_data = [item for sublist in global_data for item in sublist]
#     return np.array(global_data)

def gather_all_data_tensor(local_data_tensor):
    """
    Gathers all tensor data from all GPUs.
    """
    dist.barrier()
    world_size = dist.get_world_size()
    gathered_data = [torch.zeros_like(local_data_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_data, local_data_tensor)
    global_data = torch.cat(gathered_data, dim=0)
    return global_data

def gather_all_data_strings(local_data_strings):
    """
    Gathers all string data from all GPUs.
    """
    dist.barrier()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, local_data_strings)
    global_data = sum(gathered_data, [])
    return global_data


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

def create_model(num_classes, args):
    print("Creating model with num_classes:", num_classes)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        
        # Remove incompatible keys
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

        checkpoint_model['pos_embed'] = new_pos_embed
        model.load_state_dict(checkpoint_model, strict=False)
    return model




class DistributedWeightedSampler(Sampler):
    """
    FROM: https://github.com/pytorch/pytorch/issues/77154
    (with some modifications)

    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same
        across all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
        self,
        weights: list,
        num_samples: int = None,
        replacement: bool = True,
        num_replicas: int = None,
        rank: int = 0
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = num_replicas
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = (self.num_samples_per_replica *
                                  self.num_replicas)
        self.weights = weights
        self.replacement = replacement

        self.rank = rank

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """

        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(self.rank, self.num_replicas - 1)
            )

        weights = self.weights.copy()
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[self.rank:self.total_num_samples:self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[self.rank:self.total_num_samples:self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica


def get_weighted_loader(cls_num_list, train_dataset):
    cls_weight = 1.0 / (np.array(cls_num_list) ** args.weighted_alpha)
    cls_weight[cls_num_list == 1] = 0
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array(
        [cls_weight[train_dataset.train_labels[t]]
         for t in train_dataset.train_imgs]
    )
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    # weighted_sampler = torch.utils.data.WeightedRandomSampler(
    #     samples_weight, len(samples_weight), replacement=True
    # )
    weighted_sampler = DistributedWeightedSampler(
        samples_weight, num_samples=len(samples_weight), rank=get_rank()
    )

    weighted_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=weighted_sampler,
        num_workers=5,
        prefetch_factor=4,
        drop_last=True,
    )
    return weighted_loader


def main(args):
    init_distributed_mode(args)

    if args.restart_epoch > 0:
        assert args.restart_out_dir is not None
        assert os.path.exists(args.restart_out_dir)
        args.out_dir = args.restart_out_dir
    else:
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m-%d-%Y-%H:%M")
        args.out_dir = args.out_dir + "-" + dt_string
        print(args)
        os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device) # maybe not needed? 

    seed = args.seed + get_rank()
    torch.manual_seed(seed)

    # torch.cuda.set_device(args.gpuid)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    log = open(f"{args.out_dir}/%s.txt" % args.id, "w")
    log.flush()
    log.write(str(args) + "\n")
    log.flush()

    loader = dataloader.iNatDataLoader(
        root=args.data_path,
        batch_size=args.batch_size,
        num_workers=5,
        num_batches=args.num_batches,
        distributed = True
    )

    print("| Building net")
    net1 = create_model(num_classes = 1000, args = args)
    net2 = create_model(num_classes = 1000, args = args)
    net1 = net1.cuda(args.gpu)
    net2 = net2.cuda(args.gpu)

    net1 = torch.nn.parallel.DistributedDataParallel(
        net1, device_ids=[args.gpu], find_unused_parameters=True
        )
    net2 = torch.nn.parallel.DistributedDataParallel(
        net2, device_ids=[args.gpu], find_unused_parameters=True
        )


    cudnn.benchmark = True
    if args.restart_epoch > 0:
        net1.load_state_dict(torch.load(
                                f"{args.out_dir}/%s_net1.pth.tar" % args.id))
        net2.load_state_dict(torch.load(
                                f"{args.out_dir}/%s_net2.pth.tar" % args.id))

    optimizer1 = optim.Adam(net1.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer2 = optim.Adam(net2.parameters(), lr=args.lr, weight_decay=1e-5)

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()
    class_counts = get_class_counts(
        loader.run("eval_train", num_classes=args.num_class).dataset, args
    )

    global ldam
    ldam = LDAMLoss(class_counts)

    best_acc = [0, 0]
    for epoch in range(args.restart_epoch, args.num_epochs + 1):
        loader.sampler.set_epoch(epoch)
        lr = args.lr
        if epoch >= 40:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        if epoch < args.warmup:  # warm up
            train_loader = loader.run("warmup", num_classes=args.num_class)
            print("Warmup Net1")
            if args.use_cmo and epoch > 4:
                class_counts = get_class_counts(train_loader.dataset, args)
                train_loader = get_weighted_loader(class_counts,
                                                   train_loader.dataset)
            warmup(net1, optimizer1, train_loader, epoch, CEloss, conf_penalty)
            train_loader = loader.run("warmup", num_classes=args.num_class)
            if args.use_cmo and epoch > 4:
                class_counts = get_class_counts(train_loader.dataset, args)
                train_loader = get_weighted_loader(class_counts,
                                                   train_loader.dataset)

            train_loader.sampler.set_epoch(epoch)  # need for distributed

            print("\nWarmup Net2")
            warmup(net2, optimizer2, train_loader, epoch, CEloss, conf_penalty)
            if args.use_gnn and epoch == args.warmup - 1:
                train_gnn(net2, args, loader)

        elif epoch > args.restart_epoch:
            pred1 = prob1 > args.p_threshold  # divide dataset
            pred2 = prob2 > args.p_threshold

            print("\n\nTrain Net1")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2, paths=paths2, num_classes=args.num_class
            )  # co-divide
            if args.use_cmo:
                class_counts = get_class_counts(labeled_trainloader.dataset, args)
                weighted_labeled_loader = get_weighted_loader(
                    class_counts, labeled_trainloader.dataset
                )
                weighted_labeled_loader.sampler.set_epoch(epoch)

            train(
                epoch,
                net1,
                net2,
                optimizer1,
                labeled_trainloader,
                unlabeled_trainloader,
                CEloss,
                conf_penalty,
                cmo_loader=weighted_labeled_loader if args.use_cmo else None,
            )  # train net1
            print("\nTrain Net2")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1, paths=paths1, num_classes=args.num_class
            )  # co-divide
            train(
                epoch,
                net2,
                net1,
                optimizer2,
                labeled_trainloader,
                unlabeled_trainloader,
                CEloss,
                conf_penalty
            )  # train net2
            if (epoch % 5 == 0 or epoch == args.num_epochs) and args.use_gnn:
                train_gnn(net2, args, loader)

        val_loader = loader.run("val", num_classes=args.num_class)  # validation
        test_loader = loader.run("test", num_classes=args.num_class) 
        acc1 = val(net1, val_loader, test_loader, 1, best_acc)
        if (epoch % 5 == 0 or epoch == args.num_epochs) and args.use_gnn:
            acc2 = val_gnn(net2, CEloss, args, epoch, loader, best_acc, 2)
        else:
            acc2 = val(net2, val_loader, test_loader, 2, best_acc)
        wandb.log({'acc1': acc1, 'acc2': acc2})
        log.flush()
        if epoch >= args.warmup - 1:
            print("\n==== net 1 evaluate next epoch training data loss ====")
            eval_loader = loader.run(
                "eval_train", num_classes=args.num_class
            )  # evaluate training data loss for next epoch
            prob1, paths1 = eval_train(epoch, net1, eval_loader, CE, args)
            print("\n==== net 2 evaluate next epoch training data loss ====")
            eval_loader = loader.run("eval_train", num_classes=args.num_class)
            if args.use_gnn:
                prob2, paths2 = eval_train_gnn(net2, CE, args, epoch, loader)
            prob2, paths2 = eval_train(epoch, net2, eval_loader, CE, args)
            # Serialize prob1
            prob1 = prob1
            prob2 = prob2
            print('paths1')
            print(len(paths1))
            print('probs1')
            print(len(prob1))
            pd.DataFrame(
                {
                    "probs": prob1,
                    "paths": paths1,
                }
            ).to_csv(f"{args.out_dir}/%s_net1_probs.csv" % epoch)
            print('paths2')
            print(len(paths2))
            print('probs2')
            print(len(prob2))
            
            pd.DataFrame(
                {
                    "probs": prob2,
                    "paths": paths2,
                }
            ).to_csv(f"{args.out_dir}/%s_net2_probs.csv" % epoch)

    test_loader = loader.run("test", num_classes=args.num_class)
    net1.load_state_dict(torch.load(
                                f"{args.out_dir}/%s_net1.pth.tar" % args.id))
    net2.load_state_dict(torch.load(
                                f"{args.out_dir}/%s_net2.pth.tar" % args.id))
    if args.use_gnn:
        acc = test_with_gnn(net1, net2, args, -1, loader, test_loader)
    else:
        acc = test(net1, net2, test_loader)

    log.write("Test Accuracy:%.2f\n" % (acc))
    log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DivideMix", parents=[get_arg_parser()])

    args = parser.parse_args()
    wandb.init(
        # set the wandb project where this run will be logged
        project="inaturalist",

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "backbone",
            "dataset": "inat_c_train100k",
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        }
    )
    print("learning_rate")
    print(args.lr)
    print("batch_size")
    print(args.batch_size)

    main(args)
