'''Evaluation function'''

import numpy as np
import torch
from torch import nn
import argparse
import matplotlib.pyplot as plt
import cmocean  
import math
import torch.nn.functional as F

from data_loader import getData
from utils import *
from src.models import *
from utils import LossGenerator
import os
# % --- %
# Evaluate models
# % --- %

def load_everything(args, test1_loader, test2_loader, model, DIR="/pscratch/sd/j/junyi012/superbench_v2/eval_buffer/"):
    '''
    Load any model and save the LR,HR,Predictions as seperate .npy files to DIR

    Args:
        args (object): The arguments object containing various parameters.
        test1_loader (object): The data loader for test1.
        test2_loader (object): The data loader for test2.
        model (object): The model to be used for prediction.
        DIR (str, optional): The directory path to save the files.

    Returns:
        bool: True if the operation is successful, False otherwise.
    '''
    if args.model != 'FNO2D_patch':
        with torch.no_grad():
            lr_list, hr_list, pred_list = [], [], []
            for batch_idx, (data, target) in enumerate(test2_loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data)
                lr, hr, pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
                lr_list.append(lr)
                hr_list.append(hr)
                pred_list.append(pred)
            pred_list = np.concatenate(pred_list)
            lr_list = np.concatenate(lr_list)
            hr_list = np.concatenate(hr_list)
            np.save(DIR + f"{args.data_name}_{args.upscale_factor}_lr_{args.method}_{args.noise_ratio}.npy", lr_list)
            np.save(DIR + f"{args.data_name}_{args.upscale_factor}_hr_{args.method}_{args.noise_ratio}.npy", hr_list)
            np.save(DIR + f"{args.data_name}_{args.upscale_factor}_{args.model}_pred_{args.method}_{args.noise_ratio}.npy", pred_list)
    else:
        with torch.no_grad():
            lr_list, hr_list, pred_list = [], [], []
            for batch_idx, (data, target) in enumerate(test2_loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                hr_patch_size = 128
                hr_stride = 128
                lr_patch_size = 128 // args.upscale_factor
                lr_stride = 128 // args.upscale_factor
                lr_patches = data.unfold(2, lr_patch_size, lr_stride).unfold(3, lr_patch_size, lr_stride)
                hr_patches = target.unfold(2, hr_patch_size, hr_stride).unfold(3, hr_patch_size, hr_stride)
                if lr_patches.shape[2] != hr_patches.shape[2] or lr_patches.shape[3] != hr_patches.shape[3]:
                    print("patch size not match")
                    return False
                output = torch.zeros_like(hr_patches)
                for i in range(hr_patches.shape[2]):
                    for j in range(hr_patches.shape[3]):
                        lr = lr_patches[:, :, i, j]
                        with torch.no_grad():
                            output_p = model(lr)
                            output[:, :, i, j] = output_p
                patches_flat = output.permute(0, 1, 4, 5, 2, 3).contiguous().view(1, target.shape[1] * hr_patch_size ** 2, -1)
                output = F.fold(patches_flat, output_size=(target.shape[-2], target.shape[-1]), kernel_size=(hr_patch_size, hr_patch_size), stride=(hr_stride, hr_stride))
                lr_data, hr_data, pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
                lr_list.append(lr_data)
                hr_list.append(hr_data)
                pred_list.append(pred)
            pred_list = np.concatenate(pred_list)
            lr_list = np.concatenate(lr_list)
            hr_list = np.concatenate(hr_list)
            # if os.path.exists(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_lr.npy") == False:
            np.save(DIR + f"{args.data_name}_{args.upscale_factor}_lr_{args.method}_{args.noise_ratio}.npy", lr_list)
            np.save(DIR + f"{args.data_name}_{args.upscale_factor}_hr_{args.method}_{args.noise_ratio}.npy", hr_list)
            np.save(DIR + f"{args.data_name}_{args.upscale_factor}_{args.model}_pred_{args.method}_{args.noise_ratio}.npy", pred_list)
    return True

def get_single_pred(args, lr, hr, model, save_name, location=(3,0)):
    """
    Generate a single prediction using the specified model.

    Args:
        args (argparse.Namespace): The command-line arguments.
        lr (numpy.ndarray): The low-resolution input data.
        hr (numpy.ndarray): The high-resolution target data.
        model (torch.nn.Module): The model used for prediction.
        save_name (str): The name of the file to save the prediction.
        location (tuple, optional): The location of the batch and channel to use. Defaults to (3, 0).

    Returns:
        bool: True if the prediction is successfully generated and saved as "save_name" False otherwise.
    """

    if args.model != 'FNO2D_patch':
        batch, channel = location 
        data, target = lr[batch:batch+1], hr[batch:batch+1]
        data, target = torch.from_numpy(data).to(args.device).float(), torch.from_numpy(target).to(args.device).float()
        with torch.no_grad():
            output = model(data)
            output = output.cpu().numpy()
            if os.path.exists(save_name) == True:
                np.save(save_name,output)
            else:
                print("pred has been saved")
    else:
        batch, channel = location 
        data, target = lr[batch:batch+1], hr[batch:batch+1]
        with torch.no_grad():
            data, target = torch.from_numpy(data).to(args.device).float(), torch.from_numpy(target).to(args.device).float()
            hr_patch_size = 128
            hr_stride = 128
            lr_patch_size = 128//args.upscale_factor
            lr_stride = 128//args.upscale_factor
            lr_patches = data.unfold(2, lr_patch_size, lr_stride).unfold(3, lr_patch_size, lr_stride)
            hr_patches = target.unfold(2, hr_patch_size, hr_stride).unfold(3, hr_patch_size, hr_stride)
            if lr_patches.shape[2] != hr_patches.shape[2] or lr_patches.shape[3] != hr_patches.shape[3]:
                print("patch size not match")
                return False
            output = torch.zeros_like(hr_patches)
            for i in range(hr_patches.shape[2]):
                for j in range(hr_patches.shape[3]):
                    lr = lr_patches[:,:,i,j]
                    with torch.no_grad():
                        output_p = model(lr)
                        output[:,:,i,j] = output_p
            patches_flat = output.permute(0, 1, 4, 5, 2, 3).contiguous().view(1, hr.shape[1]*hr_patch_size**2, -1)
            # Fold the patches back
            output = F.fold(patches_flat, output_size=(hr.shape[-2], hr.shape[-1]), kernel_size=(hr_patch_size, hr_patch_size), stride=(hr_stride, hr_stride))
        output = output.cpu().numpy()
        # if os.path.exists(save_name) == False:
        np.save(save_name,output)

    return True

def validate_phyLoss(args, test1_loader, test2_loader, model):
    """
    Validates the physics loss (divergence loss) for the given model on two test loaders.

    Args:
        args (argparse.Namespace): The command-line arguments.
        test1_loader (torch.utils.data.DataLoader): The data loader for the first test set.
        test2_loader (torch.utils.data.DataLoader): The data loader for the second test set.
        model (torch.nn.Module): The model to be evaluated.

    Returns:
        Tuple[float, float]: The average physics loss for the first and second test sets, respectively.
    """
    
    avg_phyloss1, avg_phyloss2 = 0.0, 0.0

    MSEfunc = nn.MSELoss()
    lossgen = LossGenerator(args, dx=2.0*math.pi/2048.0, kernel_size=5)
    
    c = 0
    with torch.no_grad():
        for batch in test1_loader:
            input, target = batch[0].float().to(args.device), batch[1].float().to(args.device)
            model.eval()
            out = model(input)
            div = lossgen.get_div_loss(output=out)
            phy_loss = MSEfunc(div, torch.zeros_like(div).to(args.device)) # calculating physics loss
            avg_phyloss1 += phy_loss.item() * target.shape[0]
            c += target.shape[0]
    avg_phyloss1 /= c

    c = 0
    with torch.no_grad():
        for batch in test2_loader:
            input, target = batch[0].float().to(args.device), batch[1].float().to(args.device)
            model.eval()
            out = model(input)
            div = lossgen.get_div_loss(output=out)
            phy_loss = MSEfunc(div, torch.zeros_like(div).to(args.device)) # calculating physics loss
            avg_phyloss2 += phy_loss.item() * target.shape[0]
            c += target.shape[0]
    avg_phyloss2 /= c

    return avg_phyloss1, avg_phyloss2

def normalize(args,target,mean,std):
    mean = torch.Tensor(mean).to(args.device).view(1,target.shape[1],1,1)
    std = torch.Tensor(std).to(args.device).view(1,target.shape[1],1,1)
    target = (target - mean) / std
    return target

def validate_all_metrics(args, test1_loader, test2_loader, model, mean, std):
    """
    Calculates various evaluation metrics for a given model on test datasets.

    Args:
        args (argparse.Namespace): Command-line arguments.
        test1_loader (torch.utils.data.DataLoader): DataLoader for the first test dataset.
        test2_loader (torch.utils.data.DataLoader): DataLoader for the second test dataset.
        model (torch.nn.Module): Trained model to evaluate.
        mean (torch.Tensor): Mean values used for normalization.
        std (torch.Tensor): Standard deviation values used for normalization.

    Returns:
        Tuple of metric averages for the first and second test datasets:
        - Tuple of average RINE (Relative Infinite Norm Error) values for the first and second test datasets.
        - Tuple of average RFNE (Relative Frobenius Norm Error) values for the first and second test datasets.
        - Tuple of average PSNR (Peak Signal-to-Noise Ratio) values for the first and second test datasets.
        - Tuple of average SSIM (Structural Similarity Index) values for the first and second test datasets.
        - Tuple of average MSE (Mean Squared Error) values for the first and second test datasets.
        - Tuple of average MAE (Mean Absolute Error) values for the first and second test datasets.
    """
    from torchmetrics import StructuralSimilarityIndexMeasure

    ssim = StructuralSimilarityIndexMeasure().to(args.device)
    rine1, rine2, rfne1, rfne2, psnr1, psnr2, ssim1, ssim2,mse1,mse2,mae1,mae2 = [], [], [], [], [], [], [], [],[],[],[],[]
    first = True
    # Helper function for PSNR
    def compute_psnr(true, pred):
        mse = torch.mean((true - pred) ** 2)
        if mse == 0:
            return float('inf')
        max_value = torch.max(true)
        return 20 * torch.log10(max_value / torch.sqrt(mse))

    with torch.no_grad():
        for loader, (rine_list, rfne_list, psnr_list, ssim_list,mse_list,mae_list) in zip([test1_loader, test2_loader],
                                                                        [(rine1, rfne1, psnr1, ssim1,mse1,mae1),
                                                                         (rine2, rfne2, psnr2, ssim2,mse2,mae2)]):
            test = 0
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data)
                if first == True:
                    output2 = output
                    data2 = data
                    target2 = target
                    first = False
                output = normalize(args, output, mean, std)
                target = normalize(args, target, mean, std)

                # MSE 
                for i in range(target.shape[0]): # for loop for drop last
                    mse = torch.mean((target[i] - output[i]) ** 2,dim =(-1,-2,-3))
                    mse_list.append(mse.cpu())

                    # MAE
                    mae = torch.mean(torch.abs(target[i] - output[i]),dim=(-1,-2,-3))
                    mae_list.append(mae.cpu())
                    # INE
                    err_ine = torch.norm(target[i]-output[i], p=np.inf, dim=(-1, -2)) 
                    rine_list.append(err_ine.cpu())

                    # RFNE
                    err_rfne = torch.norm(target[i]-output[i], p=2, dim=(-1, -2)) / torch.norm(target[i], p=2, dim=(-1, -2))
                    rfne_list.append(err_rfne.cpu())

                # PSNR
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        err_psnr = compute_psnr(target[i, j, ...], output[i, j, ...])
                        psnr_list.append(err_psnr.cpu())

                # SSIM
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        err_ssim = ssim(target[i:(i+1), j:(j+1), ...], output[i:(i+1), j:(j+1), ...])
                        ssim_list.append(err_ssim.cpu())
            test += 1

    avg_rine1, avg_rine2 = torch.mean(torch.stack(rine1)).item(), torch.mean(torch.stack(rine2)).item()
    avg_rfne1, avg_rfne2 = torch.mean(torch.stack(rfne1)).item(), torch.mean(torch.stack(rfne2)).item()
    avg_psnr1, avg_psnr2 = torch.mean(torch.stack(psnr1)).item(), torch.mean(torch.stack(psnr2)).item()
    avg_ssim1, avg_ssim2 = torch.mean(torch.stack(ssim1)).item(), torch.mean(torch.stack(ssim2)).item()
    avg_mse1,avg_mse2 = torch.mean(torch.stack(mse1)).item(), torch.mean(torch.stack(mse2)).item()
    avg_mae1,avg_mae2 = torch.mean(torch.stack(mae1)).item(), torch.mean(torch.stack(mae2)).item()

    return (avg_rine1, avg_rine2), (avg_rfne1, avg_rfne2), (avg_psnr1, avg_psnr2), (avg_ssim1, avg_ssim2),(avg_mse1,avg_mse2),(avg_mae1,avg_mae2)

    
def main():  
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument("save_prediction", type=str,default="True" ,help="save predictions as .npy file")
    # arguments for data
    parser.add_argument('--data_name', type=str, default='nskt_16k', help='dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/nskt16000_1024', help='the folder path of dataset')
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size for high-resolution snapshots')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')

    # arguments for evaluation
    parser.add_argument('--model', type=str, default='SRCNN', help='model')
    parser.add_argument('--model_path', type=str, default=None, help='saved model')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')
    
    # arguments for training
    parser.add_argument('--epochs', type=int, default=300, help='max epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--step_size', type=int, default=100, help='step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.97, help='coefficient for scheduler')
    parser.add_argument('--phy_loss_weight', type=float, default=0.0, help='physics loss weight')
    # arguments for model
    parser.add_argument('--loss_type', type=str, default='l1', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='type of scheduler')
    parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
    parser.add_argument('--in_channels', type=int, default=1, help='num of input channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='num of hidden channels')
    parser.add_argument('--out_channels', type=int, default=1, help='num of output channels')
    parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')
    parser.add_argument('--modes', type=int, default=12, help='num of modes')
    args = parser.parse_args()
    print(args)

    # % --- %
    # Set random seed to reproduce the work
    # % --- %
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # % --- %
    # Load data
    # % --- %
    resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name)
    test1_loader, test2_loader = getData(args, args.n_patches, std=std,test=True)
    hidden = args.hidden_channels
    modes = args.modes
    # % --- %
    # Get model
    # % --- %
    upscale = args.upscale_factor
    window_size = 8
    height = (args.crop_size // upscale // window_size + 1) * window_size
    width = (args.crop_size // upscale // window_size + 1) * window_size
    if args.data_name == 'era5':
        height = (720 // upscale // window_size + 1) * window_size # for era5 
        width = (1440 // upscale // window_size + 1) * window_size # for era5
    model_list = {
            'subpixelCNN': subpixelCNN(args.in_channels, upscale_factor=args.upscale_factor, width=1, mean = mean,std = std),
            'SRCNN': SRCNN(args.in_channels, args.upscale_factor,mean,std),
            'EDSR': EDSR(args.in_channels, 64, 16, args.upscale_factor, mean, std),
            'WDSR': WDSR(args.in_channels,args.in_channels, 32,18, args.upscale_factor, mean, std),
            'SwinIR': SwinIR(upscale=args.upscale_factor, in_chans=args.in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
            'Bicubic': Bicubic(args.upscale_factor),
            "FNO2D":FNO2D(layers=[hidden, hidden, hidden, hidden, hidden],modes1=[modes, modes, modes, modes],modes2=[modes, modes, modes, modes],fc_dim=128,in_dim=args.in_channels,out_dim=args.in_channels,mean= mean,std=std,scale_factor=upscale),
    }
    # Regarding train with physics loss
    if args.model.startswith('SwinIR'):
        name = "SwinIR"
    elif args.model.startswith('FNO2D'):
        name = "FNO2D"
    else:
        name = args.model

    model = model_list[name]
    model = torch.nn.DataParallel(model)
    if args.model_path is None:
        model_path = 'results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.noise_ratio) + '_' + str(args.seed) + '.pt'
    else:
        model_path = args.model_path
    if args.model != 'Bicubic':
        model = load_checkpoint(model, model_path)
        model = model.to(args.device)

        # Model summary   
        print('**** Setup ****')
        print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')    

    else: 
        print('Using bicubic interpolation...')  

    import json
    
    # Check if the results file already exists and load it, otherwise initialize an empty list
    try:
        with open("eval_results.json", "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
        print("No results file found, initializing a new one.")
    # Create a unique key based on your parameters
    key = f"{args.model}_{args.data_name}_{args.method}_{args.upscale_factor}_{args.noise_ratio}"

# Check if the key already exists in the dictionary
    if key not in all_results:
        all_results[key] = {
            "model": args.model,
            "dataset": args.data_name,
            "method": args.method,
            "scale factor": args.upscale_factor,
            "noise ratio": args.noise_ratio,
            "parameters": (sum(p.numel() for p in model.parameters())/1000000.0),
            "metrics": {}
        }

    INE, RFNE, PSNR, SSIM,MSE,MAE = validate_all_metrics(args, test1_loader, test2_loader, model, mean, std)
    # Validate and store Infinity norm results
    # ine1, ine2 = validate_RINE(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["IN"] = {'test1 error': INE[0], 'test2 error': INE[1]}

    # Validate and store RFNE results
    # error1, error2 = validate_RFNE(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["RFNE"] = {'test1 error': RFNE[0], 'test2 error': RFNE[1]}

    # Validate and store PSNR results
    # error1, error2 = validate_PSNR(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["PSNR"] = {'test1 error': PSNR[0], 'test2 error': PSNR[1]}

    # Validate and store SSIM results
    # error1, error2 = validate_SSIM(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["SSIM"] = {'test1 error': SSIM[0], 'test2 error': SSIM[1]}
    # Validate and store MSE results
    all_results[key]["metrics"]["MSE"] = {'test1 error': MSE[0], 'test2 error': MSE[1]}
    # Validate and store MAE results
    all_results[key]["metrics"]["MAE"] = {'test1 error': MAE[0], 'test2 error': MAE[1]}
    # Validate and store Physics loss results for specific data names
    if args.data_name in ["nskt_16k", "nskt_32k"] or args.data_name.startswith("nskt_16k_sim") or args.data_name.startswith("nskt_32k_sim"):
        phy_err1, phy_err2 = validate_phyLoss(args, test1_loader, test2_loader, model)
        all_results[key]["metrics"]["Physics"] = {'test1 error': phy_err1, 'test2 error': phy_err2}

    # all_results.sorted()
    # Serialize the updated results list to the JSON file
    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    
    if args.save_prediction.lower() == "true":
        print("saving predictions as .npy file")
        load_everything(args, test1_loader, test2_loader, model)


if __name__ =='__main__':
    main()