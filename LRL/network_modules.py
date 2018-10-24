from .utils import *

class NoisyLinear(nn.Module):
    """NoisyLinear module for Noisy Net exploration technique"""
    
    def __init__(self, in_features, out_features, std_init=0.6):  # original paper: std_init = 0.4
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
