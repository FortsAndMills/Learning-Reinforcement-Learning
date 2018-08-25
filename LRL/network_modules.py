from .utils import *

#TODO: needs commentaries
class NoisyLinear(nn.Module):
    """NoisyLinear module for Noisy Net exploration technique"""
    
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(Tensor(out_features, in_features), requires_grad=True)
        self.weight_sigma = nn.Parameter(Tensor(out_features, in_features), requires_grad=True)
        
        self.bias_mu    = nn.Parameter(Tensor(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(Tensor(out_features), requires_grad=True)
        
        self.reset_parameters()
    
    def forward(self, x):
        if self.training:
            epsilon_in  = self._scaled_noise(self.in_features)
            epsilon_out = self._scaled_noise(self.out_features)
            
            weight = self.weight_mu + self.weight_sigma.mul(epsilon_out.ger(epsilon_in))            
            bias   = self.bias_mu + self.bias_sigma.mul(self._scaled_noise(self.out_features))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        # parameters initialisation
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def _scaled_noise(self, size):
        # returns scaled noise of given size
        if USE_CUDA:
            x = torch.randn(size).cuda()
        else:
            x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
        
    def magnitude(self):
        # returns summed magnitudes of noise and number of noisy parameters
        return (self.weight_sigma.abs().sum() + self.bias_sigma.abs().sum()).detach().cpu().numpy(), self.in_features*self.out_features + self.out_features
