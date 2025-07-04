#-*- coding:utf-8 -*-  
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # visible gpu <<<<<<<<<<=================
import os.path as osp
import argparse
from models.smnet_best import SMNet
from utils.utils import *
from torch.utils.data import DataLoader
import time
from dataset.training_dataset import Testing_Dataset
import warnings
import pytorch_ssim
from tqdm import tqdm
warnings.filterwarnings("ignore")


def pad_image(image, patch_size, stride_size):
    _, _, h, w = image.size()
    pad_h = (stride_size - (h-patch_size) % stride_size) % stride_size
    pad_w = (stride_size - (w-patch_size) % stride_size) % stride_size
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode='reflect')
    return padded_image


def get_args():
    parser = argparse.ArgumentParser(description="Test Setting")
    parser.add_argument("--dataset_dir", type=str, default='',
                        help='dataset directory(Kalantari|Prabhakar|Tel)')   # <<<<<<<<<<=================

    parser.add_argument('--train_patch_size', type=int, default=128,
                        help='patch size for training (default: 128)')
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
    parser.add_argument("--log_dir", type=str, default='',
                        help='log directory')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='testing batch size (default: 1)')
    parser.add_argument('--test_num_workers', type=int, default=1, metavar='N', help='number of workers to fetch data (default: 1)')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for test(default: 256)')
    parser.add_argument('--pretrained_model', type=str, default='', help='pretrained model path')  # <<<<<<<<<<<<==================
    parser.add_argument('--save_results', action='store_true', default=True, help='save output results')
    parser.add_argument('--save_dir', type=str, default="", help='save directory for output results')  #  <<<<<<<<<<<<==================
    return parser.parse_args()



def main():
    start_time = time.time()
    # Settings
    args = get_args()
    # log
    if not osp.exists(osp.dirname(args.log_dir)):
        os.makedirs(osp.dirname(args.log_dir))
        if not osp.exists(args.log_dir):
            os.makedirs(args.log_dir)

    if args.pretrained_model.endswith('.pth'):
        model_names = [args.pretrained_model]
    else:
        model_names = [os.path.join(args.pretrained_model, f) for f in os.listdir(args.pretrained_model) if f.endswith('.pth')]
    model_names.sort()
    print("Model Names: ", model_names)
    for model_name in model_names:
        # pretrained_model
        print(">>>>>>>>> Start Testing >>>>>>>>>")
        print("Load weights from: ", model_name)

        # cuda and devices
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        # model architecture
        model = SMNet(img_size=128, in_chans=6, embed_dim=60, depths=[6, 6, 6, 6],
                       num_heads=[6, 6, 6, 6], mlp_ratio=2, window_size=8)
            
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_name)['state_dict'])
        
        model.eval()

        # 测试数据
        datasets = Testing_Dataset(root_dir=args.dataset_dir, 
                                        patch_size=args.patch_size, 
                                        repeat=1, cache='none', 
                                        train_path=args.test_path, 
                                        exposure_file_name=args.exposure_file_name, 
                                        ldr_folder_name=args.ldr_folder_name, 
                                        label_file_name=args.label_file_name,
                                        mask_npy_name=args.mask_npy_name)
        test_loader = DataLoader(datasets, batch_size=1, 
                                shuffle=False, num_workers=1, 
                                pin_memory=True) 


        # metrics
        psnr_l = AverageMeter()
        ssim_l = AverageMeter()
        psnr_mu = AverageMeter()
        ssim_mu = AverageMeter()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                name = batch_data['name'][0]
                batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
                batch_ldrs = torch.cat(batch_ldrs, dim=1)
                label = batch_data['label'].to(device)
                padded_image = pad_image(batch_ldrs, args.train_patch_size, args.train_patch_size)
                pred_img = model(padded_image)
                pred_img = torch.clamp(pred_img,0,1)

                # 裁剪回原始大小
                _, _, orig_h, orig_w = label.size()
                pred_img = pred_img[:, :, :orig_h, :orig_w]  # BCHW

                # 计算指标（基于pytorch）
                mse_l =  F.mse_loss(label,pred_img)
                scene_psnr_l = (20 * torch.log10(1.0 / torch.sqrt(mse_l)))
                scene_ssim_l = pytorch_ssim.ssim(label, pred_img)

                label_mu = range_compressor(label)
                pred_img_mu = range_compressor(pred_img)
                mse_mu =  F.mse_loss(label_mu, pred_img_mu)
                scene_psnr_mu = (20 * torch.log10(1.0 / torch.sqrt(mse_mu)))
                scene_ssim_mu = pytorch_ssim.ssim(label_mu, pred_img_mu)
                
                # # update results
                psnr_l.update(scene_psnr_l)
                ssim_l.update(scene_ssim_l)
                psnr_mu.update(scene_psnr_mu)
                ssim_mu.update(scene_ssim_mu)

                # save results
                if args.save_results:
                    if not osp.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    save_path = os.path.join(args.save_dir, '{}.hdr'.format(name))
                    pred_img = pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy() 
                    pred_img = pred_img[..., ::-1]
                    print(save_path)
                    cv2.imwrite(save_path, pred_img)
                # break

            print(">>>>>>>>> Finish Testing >>>>>>>>>")
            
            print('==Test==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}'.format(
                psnr_l.avg,
                psnr_mu.avg,
                ssim_l.avg,
                ssim_mu.avg
            )) 

    endtime = time.time()
    print("Time: ", endtime - start_time)

if __name__ == '__main__':
    main()




