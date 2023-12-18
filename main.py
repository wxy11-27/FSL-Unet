import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.FSL import FSL
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(args)


def main():
    # Custom dataloader
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
        args.n_bands = 103
    elif args.dataset == 'Pavia':
        args.n_bands = 102
    elif args.dataset == 'Botswana':
        args.n_bands = 145
    elif args.dataset == 'KSC':
        args.n_bands = 176
    elif args.dataset == 'Urban':
        args.n_bands = 162
    elif args.dataset == 'IndianP':
        args.n_bands = 200
    elif args.dataset == 'Washington':
        args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
        args.n_bands = 64
    elif args.dataset == 'salinas_corrected':
        args.n_bands = 204
    elif args.dataset == 'Houston_HSI':
        args.n_bands = 144
    # Build the models
    if args.arch == 'SSFCNN':
        model = SSFCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands).cuda()
    elif args.arch == 'ConSSFCNN':
        model = ConSSFCNN(args.scale_ratio,
                          args.n_select_bands,
                          args.n_bands).cuda()
    elif args.arch == 'TFNet':
        model = TFNet(args.scale_ratio,
                      args.n_select_bands,
                      args.n_bands).cuda()
    elif args.arch == 'ResTFNet':
        model = ResTFNet(args.scale_ratio,
                         args.n_select_bands,
                         args.n_bands).cuda()
    elif args.arch == 'MSDCNN':
        model = MSDCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands).cuda()

    elif args.arch == 'SpatCNN':
        model = SpatCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands).cuda()
    elif args.arch == 'SpecCNN':
        model = SpecCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands).cuda()
    elif args.arch == 'FSL-Unet':
        model = FSL(args.arch,
                    args.scale_ratio,
                    args.n_select_bands,
                    args.n_bands).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
        .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=True)  # False
        print('Load the chekpoint of {}'.format(model_path))
        recent_rmse,recent_psnr,recent_ergas,recent_sam = validate(test_list,
                               args.arch,
                               model,
                               0,
                               args.n_epochs)
        print('rmse: ', recent_rmse,'psnr: ', recent_psnr,'ergas: ', recent_ergas,'sam: ', recent_sam)

    # Loss and Optimizer
    criterion = nn.MSELoss().cuda()

    # best_psnr = 0
    # best_rmse = 1e5
    # best_ergas = 1e5
    # best_sam = 1e5
    best_rmse,best_psnr,best_ergas,best_sam = validate(test_list,
                         args.arch,
                         model,
                         0,
                         args.n_epochs)
    print('rmse: ', best_rmse,'psnr: ', best_psnr,'ergas: ', best_ergas,'sam: ', best_sam)

    # Epochs
    print('Start Training: ')
    best_epoch = 0
    for epoch in range(args.n_epochs):
        # One epoch's traininginceptionv3
        #print('Train_Epoch_{}: '.format(epoch))
        train(train_list,
              args.image_size,
              args.scale_ratio,
              args.n_bands,
              args.arch,
              model,
              optimizer,
              criterion,
              epoch,
              args.n_epochs)

        # One epoch's validation
        print('Val_Epoch_{}: '.format(epoch))
        recent_rmse,recent_psnr,recent_ergas,recent_sam = validate(test_list,
                               args.arch,
                               model,
                               epoch,
                               args.n_epochs)
        print('rmse: ', round(recent_rmse,4),'psnr: ', round(recent_psnr,4),'ergas: ', round(recent_ergas,4),'sam: ', round(recent_sam,4))

        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
            best_epoch = epoch
            best_rmse = recent_rmse
            best_ergas = recent_ergas
            best_sam = recent_sam
            if best_psnr > 30:
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print('Saved!')
                print('')
        print('best rmse:', round(best_rmse,4), 'best psnr:', round(best_psnr,4), 'best ergas:', round(best_ergas,4),'best sam:', round(best_sam,4),'at epoch:', best_epoch)

    # print('best_psnr: ', best_psnr)


if __name__ == '__main__':
    main()
