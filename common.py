from torch import nn

class FeedForwardNet(nn.Module):
    def __init__(self, inp_dim, hidden_dim, outp_dim, n_layers, nonlinearity, dropout=0):
        super().__init__()

        layers = []
        d_in = inp_dim
        for i in range(n_layers):
            module = nn.Linear(d_in, hidden_dim)
            self.reset_parameters(module)
            layers.append(module)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            if nonlinearity == 'relu':
                nonlin = nn.ReLU(inplace=True)
            elif nonlinearity == 'tanh':
                nonlin = nn.Tanh()
            elif nonlinearity == 'elu':
                nonlin = nn.ELU(inplace=True)
            elif nonlinearity != 'none':
                raise NotImplementedError('only relu, tanh, and elu nonlinearities have been implemented')
            
            if nonlinearity != 'none':
                layers.append(nonlin)

            d_in = hidden_dim
        
        module = nn.Linear(d_in, outp_dim)
        self.reset_parameters(module)
        layers.append(module)

        self.network = nn.Sequential(*layers)

    def reset_parameters(self, module):
        init_range = 0.07
        module.weight.data.uniform_(-init_range, init_range)
        module.bias.data.zero_()
    
    def forward(self, x):
        return self.network(x)
