import numpy as np
import torch
from torch import nn
import math
def get_data_info(data_name):
    if data_name.startswith('nskt_32k'):
        resol = [2048, 2048] 
        n_fields = 3
        n_train_samples = 1000
        mean = [-1.44020306e-20,5.80499913e-20 ,-1.65496884e-15]
        std = [ 0.67831907 ,0.68145471,10.75285724]

    elif data_name.startswith('nskt_16k'):
        resol = [2048, 2048]
        n_fields = 3
        n_train_samples = 1000
        mean = [-9.48395660e-21, -7.88982956e-20 ,-2.07734654e-15]
        std=[ 0.67100703 , 0.67113945 ,10.27907989]

    elif data_name == 'cosmo':
        resol = [2048, 2048] 
        n_fields = 2
        n_train_samples = 1000
        mean = [ 3.9017, -0.3575] # [ 3.8956, -0.3664] 
        std = [0.2266, 0.4048] # [0.2191, 0.3994]

    elif data_name == 'cosmo_lres_sim'or data_name.startswith('cosmo_sim'):
        resol = [2048, 2048] 
        n_fields = 2
        n_train_samples = 1200
        mean = [3.8990, -0.3613] 
        std = [0.2237, 0.4039]  

    elif data_name == 'era5':
        resol = [720, 1440]
        n_fields = 3
        n_train_samples = 6*365
        mean = [6.3024, 278.3945, 18.4262] 
        std = [3.7376, 21.0588, 16.4687]
    else:
        raise ValueError('dataset {} not recognized'.format(data_name))

    return resol, n_fields, n_train_samples, mean, std


def set_optimizer(args, model):
    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer_type == 'AdamW':
        # swin transformer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Optimizer type {} not recognized'.format(args.optimizer_type))
    return optimizer


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def set_scheduler(args, optimizer, train_loader):
    if args.scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                    step, args.epochs * len(train_loader),
                    1,  # lr_lambda computes multiplicative factor
                    1e-6 / args.lr))  

    elif args.scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)

    elif args.scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    return scheduler


def loss_function(args):
    if args.loss_type == 'l1':
        print('Training with L1 loss...')
        criterion = nn.L1Loss().to(args.device)
    elif args.loss_type == 'l2': 
        print('Training with L2 loss...')
        criterion = nn.MSELoss().to(args.device)
    else:
        raise ValueError('Loss type {} not recognized'.format(args.loss_type))
    return criterion

def save_checkpoint(model, optimizer,save_path):
    '''save model and optimizer'''
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, save_path)


def load_checkpoint(model, save_path):
    '''load model and optimizer'''
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model loaded...')

    return model

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) // 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol    

class LossGenerator(nn.Module):
    def __init__(self, args, dx=2.0*math.pi/2048.0, kernel_size=3):
        super(LossGenerator,self).__init__()

        self.delta_x = torch.tensor(dx)

        #https://en.wikipedia.org/wiki/Finite_difference_coefficient
        self.filter_y4 = [[[[    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0],
           [1/12, -8/12,  0,  8/12, -1/12],
           [    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0]]]]

        self.filter_x4 = [[[[    0,   0,   1/12,   0,     0],
           [    0,   0,   -8/12,   0,     0],
           [    0,   0,   0,   0,     0],
           [    0,   0,   8/12,   0,     0],
           [    0,   0,   -1/12,   0,     0]]]]

        self.filter_x2 = [[[[    0,   -1/2,   0],
                    [    0,   0,   0],
                    [     0,   1/2,   0]]]]

        self.filter_y2 = [[[[    0,   0,   0],
                    [    -1/2,   0,   1/2],
                    [     0,   0,   0]]]]

        if kernel_size ==5:
            self.dx = Conv2dDerivative(
                DerFilter = self.filter_x4,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'dx_operator').to(args.device)

            self.dy = Conv2dDerivative(
                DerFilter = self.filter_y4,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'dy_operator').to(args.device)  

        elif kernel_size ==3:
            self.dx = Conv2dDerivative(
                DerFilter = self.filter_x2,
                resol = self.delta_x,
                kernel_size = 3,
                name = 'dx_operator').to(args.device)

            self.dy = Conv2dDerivative(
                DerFilter = self.filter_y2,
                resol = self.delta_x,
                kernel_size = 3,
                name = 'dy_operator').to(args.device)  

    def get_div_loss(self, output):
        '''compute divergence loss'''
        u = output[:,0:1,:,:]
        #bu,xu,yu = u.shape
        #u = u.reshape(bu,1,xu,yu)

        v = output[:,1:2,:,:]
        #bv,xv,yv = v.shape
        #v = v.reshape(bv,1,xv,yv)

        #w = output[:,0,:,:]
        u_x = self.dx(u)  
        v_y = self.dy(v)  
        # div
        div = u_x + v_y

        return div