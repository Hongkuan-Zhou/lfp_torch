import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import GlobalConfig
from lfp.model import lfp
from lfp.data import LFP_Data, pad_collate
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='lfp', help='the unique identifier for the model')
parser.add_argument('--device', type=str, default='cpu', help='device to use')
parser.add_argument('--epochs', type=int, default=50, help='number of training epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--log_dir', type=str, default='log', help='log folder')
parser.add_argument('--val_every', type=int, default=5, help='validation frequency (epochs)')
parser.add_argument('--beta', type=float, default=0.01, help='balance KL and reconstruction loss')

args = parser.parse_args()
config = GlobalConfig()
args.log_dir = os.path.join(args.log_dir, args.id)
writer = SummaryWriter(log_dir=args.log_dir)


def KL_loss(mu1, sigma1, mu2, sigma2):
    d = mu1.shape[1]
    return 0.5 * (torch.sum(torch.log(sigma2), dim=1) - torch.sum(torch.log(sigma1), dim=1) - d + torch.sum(sigma1 / sigma2, dim=1) + torch.sum((mu1 - mu2) * (mu1 - mu2) * (1.0 / sigma2), dim=1))


def get_mask(seq_l):
    N = max(seq_l)
    mask = []
    for i in seq_l:
        mask.append(np.concatenate((np.ones(i), np.zeros(N - i)), axis=0))
    return torch.unsqueeze(torch.tensor(mask), 2).to(args.device)


def compute_action_loss(gt_actions, pred_actions, seq_l):
    mask = get_mask(seq_l)
    losses = F.mse_loss(gt_actions * mask, pred_actions * mask, reduction='none').mean(dim=(1, 2))
    losses = losses / seq_l
    return losses


class Engine(object):
    """
    Engine that runs training and inference
    Args:
        - cur_epoch (int): Current epoch
        - print_evey (int): How frequently (# batches) to print loss
        - validate_every (int): How frequently (# epochs) to run validation
    """

    def __init__(self, cur_epoch=0, cur_iter=0):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        model.train()
        # Train loop
        for data in tqdm(dataloader_train):

            # efficiently zero gradients
            for p in model.parameters():
                p.grad = None
            obs = data['obs']
            acts = data['acts']
            seq_l = data['seq_l']
            ret = model(obs, acts, seq_l)
            pred_acts, mu_p, scale_p, mu_g, scale_g = ret['acts'], ret['mu_p'], ret['scale_p'], ret['mu_g'], ret['scale_g']
            kl_loss = KL_loss(mu_g, scale_g, mu_p, scale_p)
            action_loss = compute_action_loss(acts, pred_acts, seq_l)

            loss = (args.beta * kl_loss + action_loss).mean()
            loss.backward()
            loss_epoch += float(loss.item())

            num_batches += 1

            optimizer.step()
            writer.add_scalar('train_loss', loss.item(), self.cur_iter)
            self.cur_iter += 1

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        model.eval()
        with torch.no_grad():
            num_batches = 0
            loss_epoch = 0.
            for data in tqdm(dataloader_train):

                # efficiently zero gradients
                for p in model.parameters():
                    p.grad = None
                obs = data['obs']
                acts = data['acts']
                seq_l = data['seq_l']
                ret = model(obs, acts, seq_l)
                pred_acts, mu_p, scale_p, mu_g, scale_g = ret['acts'], ret['mu_p'], ret['scale_p'], ret['mu_g'], ret[
                    'scale_g']
                action_loss = compute_action_loss(acts, pred_acts, seq_l)

                loss = action_loss
                loss_epoch += float(loss.item())

                num_batches += 1

            loss = loss_epoch / float(num_batches)
            tqdm.write(f'Epoch {self.cur_epoch:03d}', f' val_loss: {loss:3.3f}')
            writer.add_scalar('val_loss', loss, self.cur_epoch)

            self.val_loss.append(loss)

    def save(self):
        save_best = False
        if self.val_loss[-1] <= self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }

        # Save the recent model/optimizer states
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.log_dir, 'recent_optim.pth'))

        # Log data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('====== Saved recent model ======>')
        if save_best:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.log_dir, 'best_optim.pth'))
            tqdm.write('====== Overwrite best model ======>')


# Data
train_set = LFP_Data(root=config.train_data, config=config)
val_set = LFP_Data(root=config.validation_data, config=config)

dataloader_train = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=pad_collate)
dataloader_val = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=pad_collate)

# Model
model = lfp(config).to(args.device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
    print('Create dir:', args.log_dir)
elif os.path.isfile(os.path.join(args.log_dir, 'recent.log')):
    print('Loading checkpoint from ' + args.log_dir)
    with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
        log_table = json.load(f)

    # Load variables
    trainer.cur_epoch = log_table['epoch']
    trainer.bestval_epoch = log_table['bestval_epoch']
    if 'iter' in log_table: trainer.cur_iter = log_table['iter']
    trainer.bestval = log_table['bestval']
    trainer.train_loss = log_table['train_loss']
    trainer.val_loss = log_table['val_loss']

    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_mode.pth')))
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'recent_optim.pth')))

# Log args
with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

for epoch in range(trainer.cur_epoch, args.epochs):
    trainer.train()
    if epoch % args.val_every == 0:
        trainer.validate()
        trainer.save()
