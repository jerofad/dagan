import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Config as cf
from torch.distributions import Categorical


# Augmentations Policy Network Module
class Policy_Controller(nn.Module):
    """ LSTM Policy Controller"""

    def __init__(self,
        operation_types = 16, 
        output_policies = cf.no_policies,
        operation_mags  = 10,
        embedding_size  = 32,
        lstm_hidden_units = 100):

        super(Policy_Controller, self).__init__()

        self._output_policies = output_policies
        self._operation_types = operation_types
        self._operation_mags  = operation_mags
        self._embedding_size  = embedding_size
        self._hidden_units    = lstm_hidden_units

        # moving average momentum for classifier losses 
        self._mu = cf.momentum

        self.emb = nn.Embedding(1+self._operation_types+self._operation_mags, 32)
        self.lstm_cell = nn.LSTMCell(self._embedding_size, self._hidden_units)

        self.fc1 = nn.Linear(self._hidden_units, self._operation_types)
        self.fc2 = nn.Linear(self._hidden_units, self._operation_mags)

        # policy_rewards are normalized loss values of classifier network
        self.register_buffer('policy_rewards', torch.zeros(self._output_policies))

        self.reset_policy_rewards()


    def reset_policy_rewards(self):

        self.policy_rewards.zero_()

    def update_policy_rewards(self, update_vals):

        # moving average update for policy rewards
        self.policy_rewards.copy_(self.policy_rewards * (1-self._mu) + self._mu * update_vals)
        return self.policy_rewards

    def forward(self, x=None, rewards_reset = False, rewards_update_vals = None , rewards_infer = False):

        if rewards_reset:
            self.reset_policy_rewards()
            return
        if rewards_update_vals is not None:
            return self.update_policy_rewards(rewards_update_vals)
        if rewards_infer:
            return self.policy_rewards

        self.logprobs = []
        self.entropy = []
        self.policies = []

        h_x = torch.randn(1, self._hidden_units).cuda()
        c_x = torch.randn(1, self._hidden_units).cuda()

        self.input_layer = self.emb(x)

        # each policy is two ops applied succesively
        for _ in range(self._output_policies): 

            h_x, c_x = self.lstm_cell(self.input_layer, (h_x, c_x))
            dist1 = Categorical(logits = self.fc1(h_x))
            op1_pred    = dist1.sample()
            op1_logprob = dist1.log_prob(op1_pred)
            op1_entropy = dist1.entropy()

            self.input_layer = self.emb(1+op1_pred)

            h_x, c_x = self.lstm_cell(self.input_layer, (h_x, c_x))
            dist2 = Categorical(logits = self.fc2(h_x))
            op1_mag_pred    = dist2.sample()
            op1_mag_logprob = dist2.log_prob(op1_mag_pred)
            op1_mag_entropy = dist2.entropy()

            self.input_layer = self.emb(self._operation_types+1+op1_mag_pred)

            h_x, c_x = self.lstm_cell(self.input_layer, (h_x, c_x))
            dist3 = Categorical(logits = self.fc1(h_x))
            op2_pred    = dist3.sample()
            op2_logprob = dist3.log_prob(op2_pred)
            op2_entropy = dist3.entropy()

            self.input_layer = self.emb(1+op2_pred)
        
            h_x, c_x = self.lstm_cell(self.input_layer, (h_x, c_x))
            dist4 = Categorical(logits = self.fc2(h_x))
            op2_mag_pred    = dist4.sample()
            op2_mag_logprob = dist4.log_prob(op2_mag_pred)
            op2_mag_entropy = dist4.entropy()

            self.input_layer = self.emb(self._operation_types+1+op2_mag_pred)

            policy_logprob = op1_logprob + op1_mag_logprob + op2_logprob + op2_mag_logprob
            policy_entropy = op1_entropy + op1_mag_entropy + op2_entropy + op2_mag_entropy
            policy = torch.tensor([op1_pred, op1_mag_pred, op2_pred, op2_mag_pred])

            self.logprobs.append(policy_logprob)
            self.entropy.append(policy_entropy)
            self.policies.append(policy)

        return torch.cat(self.logprobs, dim=-1), torch.cat(self.entropy, dim=-1), torch.cat(self.policies, dim=-1)