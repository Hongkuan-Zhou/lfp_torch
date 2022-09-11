import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class actor(nn.Module):
    """
    create actor based on observation, goal and latent space z
    """

    def __init__(self, obs_dim, act_dim, goal_dim, layer_size=1024, latent_dim=256, epsilon=1e-4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.layer_size = layer_size
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.LSTM = nn.LSTM(input_size=obs_dim + latent_dim + goal_dim, hidden_size=layer_size, num_layers=2,
                            batch_first=True)
        self.action_extractor = nn.Linear(layer_size, act_dim)

    def forward(self, o, z, g, seq_l):
        """
        :param o: observations with shape (B, T, N_o)
        :param z: latent plan embedding with shape (B, T, N_z)
        :param g: goals with shape(B, T, N_g)
        :return: actor embeddings with shape(B, T, N_a)
        """
        x = torch.concat((o, z, g), dim=-1)
        x_packed = pack_padded_sequence(x, seq_l.cpu(), batch_first=True, enforce_sorted=False)
        x_packed, _ = self.LSTM(x_packed)
        x, lens = pad_packed_sequence(x_packed, batch_first=True)
        x = self.action_extractor(x)
        return x


class plan_encoder(nn.Module):
    """
    create actor based on observation, goal and latent space z
    """

    def __init__(self, enc_in_dim, layer_size=2048, latent_dim=256, epsilon=1e-4, max_window=40):
        super().__init__()
        self.enc_in_dim = enc_in_dim
        self.layer_size = layer_size
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.max_window = max_window
        self.LSTM = nn.LSTM(input_size=enc_in_dim, hidden_size=layer_size, num_layers=2, bidirectional=True,
                            batch_first=True)
        self.z_mu = nn.Linear(layer_size * 2, latent_dim)
        self.z_scale = nn.Sequential(
            nn.Linear(layer_size * 2, latent_dim),
            nn.Softplus(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, mu, scale):
        eps = torch.randn(*mu.size()).to(self.device)
        return mu + scale * eps

    def forward(self, trajectories, seq_l):
        """
        :param trajectories: a bunch of obs and acts embedding with shape (B, T, N_o + N_a)
        :return: sampled plan encoding with shape (B, N_z) # bidirectional causes double size here
        """
        packed_trajectories = pack_padded_sequence(trajectories, seq_l.cpu(), batch_first=True, enforce_sorted=False)
        x_packed, _ = self.LSTM(packed_trajectories)
        x, lens = pad_packed_sequence(x_packed, batch_first=True)
        x_last = x[torch.arange(x.size(0)), lens-1, :]  # only return the last state to get rid of the temporal information
        mu = self.z_mu(x_last)
        scale = self.z_scale(x_last)
        return self.sample(mu, scale), mu, scale


class planner(nn.Module):
    """
    create actor based on observation, goal and latent space z
    """

    def __init__(self, obs_dim, goal_dim, layer_size=1024, latent_dim=256, epsilon=1e-4):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.layer_size = layer_size
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.mlp = nn.Sequential(
            nn.Linear(self.obs_dim + self.goal_dim, layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True)
        )
        self.z_mu = nn.Linear(layer_size, latent_dim)
        self.z_scale = nn.Sequential(
            nn.Linear(layer_size, latent_dim),
            nn.Softplus(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, mu, scale):
        eps = torch.randn(*mu.size()).to(self.device)
        return mu + scale * eps

    def forward(self, obs, goal):
        """
        :param obs: observation embedding from current frame, shape: (B, N_o)
        :param goal: goal embedding, shape: (B, N_g)
        :return: sampled plan embedding, shape: (B, N_z)
        """
        x = torch.cat((obs, goal), dim=-1)
        x = self.mlp(x)
        mu = self.z_mu(x)
        scale = self.z_scale(x + self.epsilon)
        return self.sample(mu, scale), mu, scale


class cnn(nn.Module):
    """
    create actor based on observation, goal and latent space z
    """

    def __init__(self, height=128, width=128, img_channels=3, embedding_size=64):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = img_channels

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        # TODO: change MLP size to match

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class lfp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_dim = config.obs_dim
        self.goal_dim = config.obs_dim
        self.act_dim = config.act_dim
        self.enc_in_dim = self.obs_dim + self.act_dim
        self.planner = planner(obs_dim=self.obs_dim, goal_dim=self.goal_dim)
        self.planner_encoder = plan_encoder(enc_in_dim=self.enc_in_dim)
        self.actor = actor(obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim)

    def forward(self, obs, acts, seq_l):
        """
        :param obs: the sequence of observations with the shape of (B, T, N_o)
        :param acts: the sequence of actions with the shape of (B, T, N_a)
        :param seq_l: the length of trajectories in one batch with the shape of (B,)
        :return:
        """
        _, T, _ = obs.shape
        obs_cur = obs[:, 0, :]
        goal = obs[:, -1, :]
        trajectories = torch.cat((obs, acts), dim=2)
        plan, mu_p, scale_p = self.planner(obs_cur, goal)
        gt_plan, mu_g, scale_g = self.planner_encoder(trajectories, seq_l)
        plan_tiled = torch.tile(torch.unsqueeze(plan, dim=1), (1, T, 1))
        goal_tiled = torch.tile(torch.unsqueeze(goal, dim=1), (1, T, 1))
        acts = self.actor(obs, plan_tiled, goal_tiled, seq_l)
        ret = {
            'acts': acts,
            'mu_p': mu_p,
            'scale_p': scale_p,
            'mu_g': mu_g,
            'scale_g': scale_g
        }
        return ret
