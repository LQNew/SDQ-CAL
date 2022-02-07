import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Off-Policy Actor-Critic
# And implementation of Double Q Learning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain("relu")
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)

		self.max_action = max_action
		self.apply(weight_init)

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Critic, self).__init__()
		# Q architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		self.apply(weight_init)

	def forward(self, state, action):
		sa = torch.cat([state, action], dim=1)
		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q


class SDQ(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=1,
		hidden_dim=256,
	):
		self.actor_1 = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
		self.actor_optimizer_1 = torch.optim.Adam(self.actor_1.parameters(), lr=3e-4)

		self.actor_2 = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
		self.actor_optimizer_2 = torch.optim.Adam(self.actor_2.parameters(), lr=3e-4)

		self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
		self.critic_target_1 = copy.deepcopy(self.critic_1)
		self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

		self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
		self.critic_target_2 = copy.deepcopy(self.critic_2)
		self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		action_1 = self.actor_1(state)
		action_2 = self.actor_2(state)
		critic_1_value_action_1, critic_1_value_action_2, critic_2_value_action_1, critic_2_value_action_2 = \
			self.critic_1(state, action_1), self.critic_1(state, action_2), \
			self.critic_2(state, action_1), self.critic_2(state, action_2)
		critic_action_1 = critic_1_value_action_1 + critic_2_value_action_1
		critic_action_2 = critic_1_value_action_2 + critic_2_value_action_2
		if critic_action_1 > critic_action_2:
			action = action_1
		else:
			action = action_2

		return action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample batches from replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(min=-self.noise_clip, max=self.noise_clip)
			next_action_1 = (self.actor_1(next_state) + noise).clamp(min=-self.max_action, max=self.max_action)
			next_action_2 = (self.actor_2(next_state) + noise).clamp(min=-self.max_action, max=self.max_action)

			# Compute the target Q value
			target_Q1 = self.critic_target_2(next_state, next_action_1)
			target_Q2 = self.critic_target_1(next_state, next_action_2)
			target_Q1 = reward + not_done * self.discount * target_Q1
			target_Q2 = reward + not_done * self.discount * target_Q2

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic_1(state, action), self.critic_2(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q1) + F.mse_loss(current_Q2, target_Q2)

		# Optimize the critic
		self.critic_optimizer_1.zero_grad()
		self.critic_optimizer_2.zero_grad()
		critic_loss.backward()
		self.critic_optimizer_1.step()
		self.critic_optimizer_2.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			actor_loss_1 = -self.critic_1(state, self.actor_1(state)).mean()
			actor_loss_2 = -self.critic_2(state, self.actor_2(state)).mean()

			# Optimize the actor
			self.actor_optimizer_1.zero_grad()
			self.actor_optimizer_2.zero_grad()
			actor_loss_1.backward()
			actor_loss_2.backward()
			self.actor_optimizer_1.step()
			self.actor_optimizer_2.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# save the model
	def save(self, filename):
		torch.save(self.critic_1.state_dict(), filename + "_critic_1")
		torch.save(self.critic_optimizer_1.state_dict(), filename + "_critic_optimizer_1")
		torch.save(self.critic_2.state_dict(), filename + "_critic_2")
		torch.save(self.critic_optimizer_2.state_dict(), filename + "_critic_optimizer_2")
		
		torch.save(self.actor_1.state_dict(), filename + "_actor_1")
		torch.save(self.actor_optimizer_1.state_dict(), filename + "_actor_optimizer_1")
		torch.save(self.actor_2.state_dict(), filename + "_actor_2")
		torch.save(self.actor_optimizer_2.state_dict(), filename + "_actor_optimizer_2")

	def load(self, filename):
		self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
		self.critic_optimizer_1.load_state_dict(torch.load(filename + "_critic_optimizer_1"))
		self.critic_target_1 = copy.deepcopy(self.critic_1)
		self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
		self.critic_optimizer_2.load_state_dict(torch.load(filename + "_critic_optimizer_2"))
		self.critic_target_2 = copy.deepcopy(self.critic_2)

		self.actor_1.load_state_dict(torch.load(filename + "_actor_1"))
		self.actor_optimizer_1.load_state_dict(torch.load(filename + "_actor_optimizer_1"))
		self.actor_2.load_state_dict(torch.load(filename + "_actor_2"))
		self.actor_optimizer_2.load_state_dict(torch.load(filename + "_actor_optimizer_2"))