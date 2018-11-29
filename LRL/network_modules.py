from .utils import *

# TODO: I wanted to delete that, but it somewhy works better!
def NoisyLinear(std_init=0.4):
    class NoisyLinear(nn.Module):
        """NoisyLinear module for Noisy Net exploration technique"""
        
        def __init__(self, in_features, out_features):
            super(NoisyLinear, self).__init__()
            
            self.in_features  = in_features
            self.out_features = out_features
            self.std_init     = std_init
            
            self.weight_mu    = nn.Parameter(Tensor(out_features, in_features), requires_grad=True)
            self.weight_sigma = nn.Parameter(Tensor(out_features, in_features), requires_grad=True)
            
            self.bias_mu    = nn.Parameter(Tensor(out_features), requires_grad=True)
            self.bias_sigma = nn.Parameter(Tensor(out_features), requires_grad=True)
            
            self.noise = Tensor(1)  # https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/4
            self.reset_parameters() # initialize prior noise distribution
        
        def forward(self, x):
            if self.training:
                # hack: we need to generate matrix in x out, but that's too many samples
                # instead we generate in + out noises and multiply them pairwise
                epsilon_in  = self._scaled_noise(self.in_features)
                epsilon_out = self._scaled_noise(self.out_features)
                
                weight = self.weight_mu + self.weight_sigma.mul(epsilon_out.ger(epsilon_in))            
                bias   = self.bias_mu + self.bias_sigma.mul(self._scaled_noise(self.out_features))
            else:
                weight = self.weight_mu
                bias   = self.bias_mu
            
            return F.linear(x, weight, bias)
        
        def reset_parameters(self):
            # parameters initialisation: taken from source paper
            mu_range = 1 / math.sqrt(self.weight_mu.size(1))
            
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        
        def _scaled_noise(self, size):
            # returns scaled noise of given size
            sampled_noise = self.noise.repeat(size).normal_()
            return sampled_noise.sign().mul(sampled_noise.abs().sqrt())
            
        def magnitude(self):
            # returns summed magnitudes of noise and number of noisy parameters
            return (self.weight_sigma.abs().sum() + self.bias_sigma.abs().sum()).detach().cpu().numpy(), self.in_features*self.out_features + self.out_features
    return NoisyLinear

def orthogonal_with_zero_bias_init(module, gain=1):
    class orthogonal_with_zero_bias_init(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            nn.init.orthogonal_(self.weight.data, gain=gain)
            nn.init.constant_(self.bias.data, 0)
    return orthogonal_with_zero_bias_init

# MORE BEAUTIFUL CODE, but works worse. Initialization is in question!
# This code's author says nn.Linear initialisation is ok here... 
class NoisyLinearWTF(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    """
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__(in_features, out_features)
        
        sigma_init = std_init / math.sqrt(in_features)
        self.n_params = in_features*out_features + out_features
        
        self.sigma_weight = nn.Parameter(Tensor(out_features, in_features).fill_(std_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features).to(device))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1).to(device))
        
        self.sigma_bias = nn.Parameter(Tensor(out_features).fill_(std_init))

    def forward(self, x):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

        scale = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = scale(self.epsilon_input)
        eps_out = scale(self.epsilon_output)
        
        return F.linear(x, self.weight + self.sigma_weight * torch.mul(eps_in, eps_out), self.bias + self.sigma_bias * eps_out.t())
        
    def magnitude(self):
        # returns summed magnitudes of noise and number of noisy parameters
        return (self.sigma_weight.abs().sum() + self.sigma_bias.abs().sum()).detach().cpu().numpy(), self.n_params
