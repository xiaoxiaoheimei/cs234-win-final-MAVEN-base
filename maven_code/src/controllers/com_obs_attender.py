####################################
#File: com_obs_attender.py         #
#Author: Liu Yang                  #
#Email: liuyeung@stanford.edu      #
###################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComObsAttender(nn.Module):
   def __init__(self, env):
       super(ComObsAttender, self).__init__()
       info = env.get_env_info()
       self.n_agents = info['n_agents']
       self.obs_size = info['obs_shape']
       self.n_enemies = env.n_enemies
       self.nf_al = 4 + env.unit_type_bits
       self.nf_en = 4 + env.unit_type_bits

       if env.obs_all_health:
          self.nf_al += 1 + env.shield_bits_ally
          self.nf_en += 1 + env.shield_bits_enemy

       if env.obs_last_action:
          self.nf_al += env.n_actions

       self.nf_own = env.unit_type_bits
       if env.obs_own_health:
          self.nf_own += 1 + env.shield_bits_ally

       self.move_feats_len = env.n_actions_move
       if env.obs_pathing_grid:
          self.move_feats_len += env.n_obs_pathing
       if env.obs_terrain_height:
          self.move_feats_len += env.n_obs_height
       
       self.al_offset = self.nf_own + self.n_enemies*self.nf_en #index of the visible field of the first agent

       #set index buffer
       self.al_idx = torch.LongTensor([[j for j in range(self.n_agents) if j != i] for i in range(self.n_agents)]).flatten().cuda()
       self.register_buffer('al_idx_att', self.al_idx)
       self.al_vis_idx = torch.LongTensor([self.al_offset + j*self.nf_al for j in range(self.n_agents-1)]).cuda()
       self.register_buffer('al_vis_idx_att', self.al_vis_idx)
       self.c_att = nn.Linear(self.obs_size, 3*self.obs_size) #produce tuple (query, key, value)

   def getObsSize(self):
       return 2*self.obs_size

   def forward(self, obs):
       '''
       Args:
         obs (Tensor) input tensor. (batch, n_agents, obs_size)
       ''' 
       al_visible_mask = obs[..., self.al_vis_idx] == 1#(batch, n_agents, n_agents-1)
       batch, _, _ = obs.size()
       query, key, value = self.c_att(obs).split(self.obs_size, -1) #(batch, n_agents, obs_size) for query, key, value
       key_t = torch.index_select(key, -2, self.al_idx) #(batch, n_agents*(n_agents-1), obs_size)
       key_t = key_t.view(batch, self.n_agents, self.n_agents-1, -1)
       value_t = torch.index_select(value, -2, self.al_idx)
       value_t = value_t.view(batch, self.n_agents, self.n_agents-1, -1) #(batch, n_agents, n_agents-1, obs_size)
       query_t = query.unsqueeze(-2) #(batch, n_agent, 1, obs_size)
       att_w = (query_t * key_t).sum(-1).masked_fill(al_visible_mask == 0, -9999.) #(batch, n_agents, n_agents-1)
       att_w = F.softmax(att_w, -1) #(batch, seq_len, n_agents, n_agents-1)
       att_w = (att_w * al_visible_mask.float()).unsqueeze(-1) #(batch, n_agents, n_agents-1, 1)
       env_obs = (value_t * att_w).sum(-2) #(batch, n_agents, obs_size)
       aug_obs = torch.cat([obs, env_obs], -1) #(batch, n_agents, 2*obs_size)
       return aug_obs
