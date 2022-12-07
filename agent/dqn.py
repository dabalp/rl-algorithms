# DQN agent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35 # dm_control
        # self.repr_dim = 32 * 25 * 25 # minigrid partial obs
        self.repr_dim = 32 * 17 * 17  # minigrid full
        # self.repr_dim = 32 * 38 * 38  # atari

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            # nn.Conv2d(9, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        # obs = obs.permute(
        #     0, 3, 1, 2
        # )  # permute can swap all dimensions while transpose only 2
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        h = h.reshape(-1, self.repr_dim)
        return h


class QNetEncoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, action_dim):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35 # dm_control
        # self.repr_dim = 32 * 25 * 25 # minigrid partial obs
        self.repr_dim = 32 * 17 * 17  # minigrid full
        # self.repr_dim = 32 * 39 * 39  # atari

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0
        h = self.convnet(obs)
        return h


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(
        self,
        name,
        obs_type,
        obs_shape,
        action_shape,
        hidden_dim,
        lr,
        gamma,
        batch_size,
        tau,
        update_every_steps,
        device,
        use_wandb,
        nstep,
        epsilon,
    ):
        self.name = name
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every_steps = update_every_steps
        self.device = device
        self.use_wandb = use_wandb
        self.epsilon = epsilon

        # if obs_type == "pixels":
        #     self.encoder = Encoder(obs_shape).to(self.device)
        #     self.obs_dim = self.encoder.repr_dim

        # else:
        #     self.encoder = nn.Identity()
        #     self.obs_dim = obs_shape[0]

        self.q_net = QNetEncoder(self.obs_shape, self.hidden_dim, self.action_dim).to(
            self.device
        )
        self.target_net = QNetEncoder(
            self.obs_shape, self.hidden_dim, self.action_dim
        ).to(self.device)
        # self.q_net = QNet(self.obs_dim, self.hidden_dim, self.action_dim).to(
        #     self.device
        # )
        # self.target_net = QNet(self.obs_dim, self.hidden_dim, self.action_dim).to(
        #     self.device
        # )
        self.target_net.load_state_dict(self.q_net.state_dict())

        # optimizers
        self.q_net_optim = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        # if obs_type == "pixels":
        #     self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        # else:
        #     self.encoder_optim = None

        self.train()
        self.target_net.train()

    def train(self, training=True):
        self.training = training
        # self.encoder.train(training)
        self.q_net.train(training)

    def act(self, obs, step):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if np.random.rand() > self.epsilon:
            with torch.no_grad():  # probably don't need this as it is done before act
                # q_values = self.q_net(self.encoder(obs))
                q_values = self.q_net(obs)
            action = q_values.argmax().item()
        else:
            action = np.random.randint(self.action_dim)

        return action

    def learn(self, obs, actions, rewards, discount, next_obs, step):
        metrics = dict()
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
        # Update Q network
        # q_values = self.q_net(self.encoder(obs))
        q_values = self.q_net(obs)
        # we unsqueeze(-1) the actions to get shape (batch_size, 1) which matchtes
        # rewards shape of (batch_size, 1). Unsqueeze is not required and alternatively
        # we can make sure rewards to be shape of (batch_size)
        q_values = q_values.gather(1, actions.unsqueeze(-1))

        with torch.no_grad():
            # next_q_values = self.target_net(self.encoder(next_obs))
            next_q_values = self.target_net(next_obs)
            # next_q_values shape is [batch_size, action_dim], getting max(1)[0} will
            # give shape of [batch_size], hence unsqueeze(-1) to get [batch_size, 1]
            # which will match the shape rewards and q_values
            # next_q_values = next_q_values.max(1)[0].view(self.batch_size, 1)
            next_q_values = next_q_values.max(1)[0].unsqueeze(-1)
            # next_q_values = next_q_values.max(1)[0]

            # discount will be zero for terminal states, so we don't need to worry to do
            # any masking
            next_q_values = rewards + self.gamma * discount * next_q_values

        # print(q_values.shape, next_q_values.shape)
        q_loss = F.mse_loss(q_values, next_q_values)

        if self.use_wandb:
            metrics["q_loss"] = q_loss.item()
            metrics["q_val"] = q_values.mean().item()
            metrics["q_target"] = next_q_values.mean().item()

        # if self.encoder_optim is not None:
        #     self.encoder_optim.zero_grad()
        self.q_net_optim.zero_grad()
        q_loss.backward()
        self.q_net_optim.step()
        # if self.encoder_optim is not None:
        #     self.encoder_optim.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, actions, rewards, discount, next_obs = utils.to_torch(batch, self.device)
        actions = actions[:, 0]  # need to fix this in the replay buffer or wrapper

        # Update Q network
        metrics.update(self.learn(obs, actions, rewards, discount, next_obs, step))

        # Update target network
        with torch.no_grad():
            utils.soft_update_params(self.q_net, self.target_net, self.tau)

        return metrics
