# -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.training_dataset import Training_Dataset,Validing_Dataset,Testing_Dataset
from loss.loss import Loss
from models.smnet_best import SMNet
from utils.utils import * 
from lr_scheduler.mylr import MyLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_ssim

#### stageI

def pad_image(image, patch_size, stride_size):
    _, _, h, w = image.size()
    pad_h = (stride_size - (h-patch_size) % stride_size) % stride_size
    pad_w = (stride_size - (w-patch_size) % stride_size) % stride_size
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode='reflect')
    return padded_image

def get_args():
    parser = argparse.ArgumentParser(description='All Settings',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset Settings
    parser.add_argument("--dataset_dir", type=str, default='',
                        help='dataset directory')  # <<<<<====================
    parser.add_argument('--train_path', type=str, default='Training',
                        help='train path(default: Training)')
    parser.add_argument('--test_path', type=str, default='Test',
                        help='test path(default: Test)')
    parser.add_argument('--exposure_file_name', type=str, default='exposure.txt',
                        help='exposure file name(default: exposure.txt)')
    parser.add_argument('--ldr_folder_name', type=str, default=None,
                        help='ldr folder name(default: None)')
    parser.add_argument('--label_file_name', type=str, default='HDRImg.hdr',
                        help='label file name(default: HDRImg.hdr)')
    parser.add_argument('--mask_npy_name', type=str, default='masks.npy',
                        help='mask npy name(default: masks.npy)')
    
    # Training and Test Settings
    parser.add_argument('--train_patch_size', type=int, default=128,
                        help='patch size for training (default: 128)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='patch size for test (default: 256)')
    parser.add_argument('--repeat', type=int, default=100,
                        help='number of repeat for training dataset (default: 100)')
    parser.add_argument('--logdir', type=str, default='',
                        help='target log directory')  # Log_path <<<========
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data for training (default: 8)')
    parser.add_argument('--test_num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data for test (default: 1)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--phase1_epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train without mask (default: 300)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='training batch size')  # Batch <<<========
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--resume', type=str, default=None,
                        help='load model from a .pth file (default: None)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--cache_choice', type=int, default = 1,
                        help='cache for dataloader(0: none, 1: bin, 2: in_memory)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--lr_decay', action='store_true', default=True,
                        help='learning rate decay or not')
    parser.add_argument('--le_lambda', type=float, default=0.005, metavar='N',
                        help='weight of the local-enhanced loss(default: 0.005)')
    
    # Mask Settings
    parser.add_argument('--is_mask', action='store_true', default=True,
                        help='mask in train or not')
    parser.add_argument('--is_curriculum', action='store_true', default=True,
                        help='curriculum learning or not')
    parser.add_argument('--mask_choice', type=int, default=1,
                        help='how to mask the feature(0: Random, 1: SAM guided)')
    parser.add_argument('--adl_drop_rate', type=float, default=0.75, metavar='N',
                        help='the probability of masking(default: 0.75)')
    parser.add_argument('--metric_guided', type=str, default='psnr_mu',
                        help='metric for computing local reconstruction(default: psnr_mu)')
    parser.add_argument('--is_guided_mask', type=int, default=1,
                        help='is guided mask or not(0: no, 1: yes)')
    parser.add_argument('--is_guided_loss', type=int, default=1,
                        help='is guided loss or not(0: no, 1: yes)')
    parser.add_argument('--mask_temperature', type=float, default=0.05, metavar='N',
                        help='the temperature of SAM guided mask(default: 0.1)')
    parser.add_argument('--loss_temperature', type=float, default=-0.1, metavar='N',
                        help='the possibility of SAM guided loss(default: -0.1)')
    parser.add_argument('--metric_guided_interval', type=int, default=5, metavar='N',
                        help='the interval of the metric guided before masked training(default: 5)')
    
    # Other Settings
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--is_freeze', action='store_true', default=False,
                        help='freeze partial parameters or not')
    return parser.parse_args()

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, batch_data in enumerate(tqdm(train_loader)):
        # dataloader
        batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
        batch_ldrs = torch.cat(batch_ldrs, dim=1)
        label = batch_data['label'].to(device)
        masks = batch_data['masks'].to(device)

        # calculate mask map
        if args.is_mask == True and args.mask_ratio > 0:
            mask_map = calculate_mask_map(args, label, masks)
        else:
            mask_map = None

        # calculate loss map
        loss_map = calculate_loss_map(args, label, masks)
        
        # inference
        pred = model(batch_ldrs, mask_map)

        # loss
        loss, loss_dict = criterion(pred, label, loss_map)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # batch_time.update(time.time() - end)
        # end = time.time()

        if batch_idx % args.log_interval == 0:
            logger_train.info('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'L1 Loss: {:.6f}\t'
                  'Local-Enhanced Loss: {:.6f}'.format(
                epoch,
                batch_idx * args.batch_size,
                len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item(),
                loss_dict['loss_recon'].item(),
                loss_dict['loss_le'].item(),
            ))

            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred)
            test_img = torch.cat([batch_ldrs[:,6:9,:,:], pred_img_mu, label_mu], dim=3)
            tb_figure.add_image(f'train/{batch_idx}',test_img, epoch, dataformats='NCHW')
            tb_figure.add_image('train/loss_map', loss_map, batch_idx+(epoch-1)*len(train_loader.dataset), dataformats='NCHW')
            tb_writer.add_scalar('train/loss', loss.item(), batch_idx+(epoch-1)*len(train_loader.dataset))
            tb_writer.add_scalar('train/loss_recon', loss_dict['loss_recon'].item(), batch_idx+(epoch-1)*len(train_loader.dataset))
            tb_writer.add_scalar('train/loss_le', loss_dict['loss_le'].item(), batch_idx+(epoch-1)*len(train_loader.dataset))

# for evaluation with limited GPU memory
def test_single_img(args, model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False)
    with torch.no_grad():
        for batch_data in dataloader:
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            output = model(batch_ldrs)
            img_dataset.update_result(torch.squeeze(output.detach().cpu()).numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    return pred, label

def test(args, model, device, optimizer, lr_scheduler, epoch, test_loader, ckpt_dir):
    model.eval()
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            label = batch_data['label'].to(device)

            padded_image = pad_image(batch_ldrs, args.train_patch_size, args.train_patch_size)
            pred_img = model(padded_image)
            # 裁剪回原始大小
            _, _, orig_h, orig_w = label.size()
            pred_img = pred_img[:, :, :orig_h, :orig_w]
            pred_img = torch.clamp(pred_img,0,1)

            # 计算指标（基于pytorch）
            mse_l =  F.mse_loss(label,pred_img)
            scene_psnr_l = (20 * torch.log10(1.0 / torch.sqrt(mse_l)))
            scene_ssim_l = pytorch_ssim.ssim(label, pred_img)

            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            mse_mu =  F.mse_loss(label_mu, pred_img_mu)
            scene_psnr_mu = (20 * torch.log10(1.0 / torch.sqrt(mse_mu)))
            scene_ssim_mu = pytorch_ssim.ssim(label_mu, pred_img_mu)

            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu) 

            if batch_idx % 5 == 0:
                test_img = torch.cat([batch_ldrs[:,6:9,:,:], pred_img_mu, label_mu], dim=3)
                tb_figure.add_image(f'test/{batch_idx}',test_img, epoch, dataformats='NCHW')

    if best_metric['psnr_l']['value'] < psnr_l.avg:
        best_metric['psnr_l']['value'] = psnr_l.avg
        best_metric['psnr_l']['epoch'] = epoch
    if best_metric['psnr_mu']['value'] < psnr_mu.avg:
        best_metric['psnr_mu']['value'] = psnr_mu.avg
        best_metric['psnr_mu']['epoch'] = epoch
    if best_metric['ssim_l']['value'] < ssim_l.avg:
        best_metric['ssim_l']['value'] = ssim_l.avg
        best_metric['ssim_l']['epoch'] = epoch
    if best_metric['ssim_mu']['value'] < ssim_mu.avg:
        best_metric['ssim_mu']['value'] = ssim_mu.avg
        best_metric['ssim_mu']['epoch'] = epoch

    logger_train.info('Epoch:' + str(epoch) + '\tmask_ratio :  '+ str(args.mask_ratio))
    logger_train.info('Test set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}\n'.format(
        psnr_l.avg,
        psnr_mu.avg,
        ssim_l.avg,
        ssim_mu.avg
        ))
    logger_valid.info('==Best==\tPSNR_l: {:.4f}/epoch: {}\t PSNR_mu: {:.4f}/epoch: {} \t SSIM_l: {:.4f}/epoch: {}\t SSIM_mu: {:.4f}/epoch: {}'.format(
        best_metric['psnr_l']['value'], best_metric['psnr_l']['epoch'],
        best_metric['psnr_mu']['value'], best_metric['psnr_mu']['epoch'],
        best_metric['ssim_l']['value'], best_metric['ssim_l']['epoch'],
        best_metric['ssim_mu']['value'], best_metric['ssim_mu']['epoch']
    ))

    # save_model
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(save_dict, os.path.join(args.logdir, 'epoch_{:d}_mask_ratio_{:.2f}.pth'.format(epoch, args.mask_ratio)))

    tb_writer.add_scalar('test/psnr_l', psnr_l.avg, epoch)
    tb_writer.add_scalar('test/psnr_mu', psnr_mu.avg, epoch)
    tb_writer.add_scalar('test/ssim_l', ssim_l.avg, epoch)
    tb_writer.add_scalar('test/ssim_mu', ssim_mu.avg, epoch)


def metric_guided_ratio(args, model, device, valid_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(valid_loader)):
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            label = batch_data['label'].to(device)
            masks = batch_data['masks'].to(device)

            # 找到masks中所有可能的值
            mask_id_list = torch.unique(masks)

            padded_image = pad_image(batch_ldrs, args.train_patch_size//8, args.train_patch_size//8)
            pred_img = model(padded_image)

            # 裁剪回原始大小
            _, _, orig_h, orig_w = label.size()
            pred_img = pred_img[:, :, :orig_h, :orig_w]


            # 计算指标（基于pytorch）
            if args.metric_guided == 'psnr_l':
                B, C, H, W = label.shape
                d = (pred_img - label) ** 2
                psnr_dict = {}
                for mid in mask_id_list:
                    mid = mid.item()
                    mask = (masks == mid)
                    mse = torch.sum(d[:, :, mask]) / (C * torch.sum(mask))
                    psnr_dict[mid] = 20 * torch.log10(1.0 / torch.sqrt(mse))

            elif args.metric_guided == 'psnr_mu':
                label_mu = range_compressor(label)
                pred_img_mu = range_compressor(pred_img)
                B, C, H, W = label.shape
                d = (pred_img_mu - label_mu) ** 2
                psnr_dict = {}
                for mid in mask_id_list:
                    mid = mid.item()
                    mask = (masks == mid)
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mse = torch.sum(torch.masked_select(d, mask)/ (C * torch.sum(mask)))
                    psnr_dict[mid] = 20 * torch.log10(1.0 / torch.sqrt(mse))

            if args.is_guided_mask:
                mask_guided = {k: torch.exp(v * args.mask_temperature) for k, v in psnr_dict.items()}
                total = sum(mask_guided.values())
                mask_guided = {k: v/total for k, v in mask_guided.items()}
                for mid in mask_id_list:
                    mid = mid.item()
                    args.mask_ratio_dict[mid] = args.mask_ratio * mask_guided[mid] * len(mask_id_list)
            else:
                for mid in mask_id_list:
                    mid = mid.item()
                    args.mask_ratio_dict[mid] = args.mask_ratio
            
            if args.is_guided_loss:
                loss_guided = {k: torch.exp(v * args.loss_temperature) for k, v in psnr_dict.items()}
                total = sum(loss_guided.values())
                loss_guided = {k: v/total for k, v in loss_guided.items()}
                for mid in mask_id_list:
                    mid = mid.item()
                    args.loss_ratio_dict[mid] = loss_guided[mid] * len(mask_id_list)
            else:
                for mid in mask_id_list:
                    mid = mid.item()
                    args.loss_ratio_dict[mid] = 1.0


def init_ratio(args):
    ## 修改为统计整张Mask
    scenes_dir = os.path.join(args.dataset_dir, args.train_path)  # /Kalantari/Training
    scenes_list = sorted(os.listdir(scenes_dir))
    for scene in range(len(scenes_list)):
        print(f"init ratio {scene}")
        mask_npy_path = os.path.join(os.path.join(scenes_dir, scenes_list[scene]), args.mask_npy_name)
        masks = np.load(mask_npy_path, allow_pickle=True)
        # 找到masks中所有可能的值
        mask_id_list = np.unique(masks)
        # 掩码率初始化为0
        for mid in mask_id_list:
            args.mask_ratio_dict[mid] = 0.0
        # loss权重初始化为1
        for mid in mask_id_list:
            args.loss_ratio_dict[mid] = 1.0


def main():
    ### 初始化设置 ---------------------------------------------------------------------------- ###
    print('===> Init settings')
    # settings
    args = get_args()
    
    # random seed
    if args.seed is not None:
        set_random_seed(args.seed)
    
    # logdir
    logdir = args.logdir
    tensorboard_dir_curve = os.path.join(logdir, 'tensorboard','curve')
    tensorboard_dir_figure = os.path.join(logdir, 'tensorboard','figure')
    ckpt_dir = os.path.join(logdir, 'ckpt')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(tensorboard_dir_curve):
        os.makedirs(tensorboard_dir_curve)
    if not os.path.exists(tensorboard_dir_figure):
        os.makedirs(tensorboard_dir_figure)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global logger_train
    logger_train = get_logger('train', logdir)
    global logger_valid
    logger_valid = get_logger('valid', logdir)
    global tb_writer
    tb_writer = SummaryWriter(os.path.join(tensorboard_dir_curve))
    global tb_figure
    tb_figure = SummaryWriter(os.path.join(tensorboard_dir_figure))

    # 记录参数设置
    args_dict = vars(args)
    for key, value in args_dict.items():
        logger_train.info(f'{key}: {value}')

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')


    ### 加载数据集 ---------------------------------------------------------------------------- ###
    # dataset
    print('===> Loading datasets')
    if args.cache_choice == 0:
        cache = 'none'
        print('===> No cache')
    elif args.cache_choice == 1:
        cache = 'bin'
        print('===> Cache bin')
    elif args.cache_choice == 2:
        cache = 'in_memory'
        print('===> Cache in_memory')

    # 训练数据
    train_dataset = Training_Dataset(root_dir=args.dataset_dir, 
                                     patch_size=args.train_patch_size, 
                                     repeat=args.repeat, cache=cache, 
                                     train_path=args.train_path, 
                                     exposure_file_name=args.exposure_file_name, 
                                     ldr_folder_name=args.ldr_folder_name, 
                                     label_file_name=args.label_file_name,
                                     mask_npy_name=args.mask_npy_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, 
                              pin_memory=True)  
    trainset_size = len(train_loader.dataset)

    # 计算PSNR数据
    valid_dataset = Validing_Dataset(root_dir=args.dataset_dir, 
                                    patch_size=args.patch_size, 
                                    repeat=1, cache=cache, 
                                    train_path=args.train_path, 
                                    exposure_file_name=args.exposure_file_name, 
                                    ldr_folder_name=args.ldr_folder_name, 
                                    label_file_name=args.label_file_name,
                                    mask_npy_name=args.mask_npy_name)
    valid_loader = DataLoader(valid_dataset, batch_size=1, 
                              shuffle=False, num_workers=1, 
                              pin_memory=True) 
    validset_size = len(valid_loader.dataset)
    
    # 测试数据
    test_dataset = Testing_Dataset(root_dir=args.dataset_dir, 
                                    patch_size=args.patch_size, 
                                    repeat=1, cache=cache, 
                                    train_path=args.test_path, 
                                    exposure_file_name=args.exposure_file_name, 
                                    ldr_folder_name=args.ldr_folder_name, 
                                    label_file_name=args.label_file_name,
                                    mask_npy_name=args.mask_npy_name)
    test_loader = DataLoader(test_dataset, batch_size=1, 
                              shuffle=False, num_workers=1, 
                              pin_memory=True) 
    testset_size = len(test_loader.dataset)

    print('===> Training dataset size: {}, Validing dataset size: {},Testing dataset size: {}.'.format(trainset_size, validset_size, testset_size))


    ### 初始化模型 ---------------------------------------------------------------------------- ###
    model = SMNet(img_size=128, in_chans=6, embed_dim=60, depths=[6, 6, 6, 6], 
                  num_heads=[6, 6, 6, 6], mlp_ratio=2, window_size=8,
                  is_mask=args.is_mask, adl_drop_rate=args.adl_drop_rate)
    # init
    if args.init_weights:
        init_weights(model, init_type='normal', gain=0.02)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)# 1e-8
    # lr_scheduler
    if args.lr_decay:
        lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch = args.phase1_epochs, eta_min=5e-5)

    model.to(device)
    model = nn.DataParallel(model)

    # load checkpoint
    if args.resume and os.path.isfile(args.resume):
        if args.is_freeze:
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 1
            model.load_state_dict(checkpoint['state_dict'])
            model = freeze_model(model=model, not_freeze_list=['module.conv_first.0.weight', 'module.conv_first.0.bias'])
            # optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                         lr=args.lr, betas=(0.9, 0.999), eps=1e-8)# 1e-8
            if args.lr_decay:
                lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch = args.phase1_epochs, 
                                    eta_min=5e-5)
            print("===> Start fine-tuning.")
        else:
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
    else:
        print("===> No checkpoint is founded.")


    ### 初始化变量 ---------------------------------------------------------------------------- ###
    # loss  
    criterion = Loss(le_lambda = args.le_lambda).to(device)
    # metrics
    global best_metric
    best_metric = {'psnr_l': {'value': 0., 'epoch': 0},
                   'psnr_mu': {'value': 0., 'epoch': 0},
                   'ssim_l': {'value': 0., 'epoch': 0},
                   'ssim_mu': {'value': 0., 'epoch': 0}}
    # SAM guided mask&loss
    args.mask_ratio_dict, args.loss_ratio_dict = {}, {}

    # init mask ratio and loss ratio
    print("===> init ratio.")
    init_ratio(args)


    ### 训练过程 ---------------------------------------------------------------------------- ###
    for epoch in range(args.start_epoch, args.phase1_epochs + 1):
        # 计算掩码率，最后要到1吗？
        cal_mask_ratio(args, epoch)
        logger_train.info(f'===> Epoch: {epoch}/{args.phase1_epochs} | Mask ratio: {args.mask_ratio}')
        tb_writer.add_scalar('train/mask_ratio', args.mask_ratio, epoch)
        print(f'===> Epoch: {epoch}/{args.phase1_epochs} | Mask ratio: {args.mask_ratio}')

        # 学习率
        for param_group in optimizer.param_groups:
            logger_train.info("Learning rate is: [{:1.7f}] ==".format(param_group['lr']))
            tb_writer.add_scalar('train/lr', param_group['lr'], epoch)
            print("Learning rate is: [{:1.7f}] ==".format(param_group['lr']))

        # 更新当前的掩码率和loss权重,基于PSNR加权
        if epoch > args.phase1_epochs or (epoch > 0 and epoch % args.metric_guided_interval == 0):
            if args.is_guided_mask or args.is_guided_loss:
                print('===> Metric guided')
                metric_guided_ratio(args, model, device, valid_loader)
                print('===> Finshed')

        # 训练
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        if args.lr_decay:
            lr_scheduler.step()

        # 测试
        print(f"==> start test of epoch {epoch}.")
        test(args, model, device, optimizer, lr_scheduler, epoch, test_loader, ckpt_dir)


if __name__ == '__main__':
    main()

