import cv2
import torch
import numpy as np
from PIL import Image
import glob
import os
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

mask_id = 0

def sam_preprocess(folder_name, mask_generator):
    global mask_id
    print(folder_name)
    tif_names = sorted(glob.glob(os.path.join(folder_name, "*.tif")))
    image = cv2.imread(tif_names[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    # 按照area从小到大排序，如果和之前的segmentation有重叠，就不要了
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    # 只保存segmentation
    mask_id += 1
    min_mask_id = mask_id
    mask_map = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * mask_id
    for i, mask in enumerate(masks):
        segmentation = mask['segmentation']
        mask_id += 1
        mask_map[segmentation] = mask_id
    count = min_mask_id
    for i in range(min_mask_id, mask_id+1):
        m = mask_map==i
        mask_map[m] = count
        if np.sum(m) < 1000:
            mask_map[m] = min_mask_id
        else:
            count += 1
    mask_id = count - 1
    max_mask_id = mask_id
    mask_image = mask_map.copy()
    mask_image = 255. * (mask_image-min_mask_id)/(max_mask_id-min_mask_id)
    # H,W -> H,W,1
    mask_map = mask_map[..., None]
    # 存到PIL Image中 
    mask_image = mask_image.astype(np.uint8)
    mask_image = Image.fromarray(mask_image)
    np.save(os.path.join(folder_name, "masks.npy"), mask_map)
    mask_image.save(os.path.join(folder_name, "masks.png"))

def random_preprocess(folder_name):
    global mask_id
    mask_id = 0
    print(folder_name)
    tif_names = sorted(glob.glob(os.path.join(folder_name, "*.tif")))
    image = cv2.imread(tif_names[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_id += 1
    mask_map = np.ones((image.shape[0], image.shape[1], 1), dtype=np.float32) * mask_id
    # 存到PIL Image中
    mask_image = mask_map.astype(np.uint8)
    mask_image = Image.fromarray(mask_image)
    np.save(os.path.join(folder_name, "masks.npy"), mask_map)
    mask_image.save(os.path.join(folder_name, "masks.png"))
    return mask_map

def main():
    parser = argparse.ArgumentParser(description='SAM pre_mask',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, default=''
                        , help="dataset dir")  # <<<===========================
    parser.add_argument('--train_path', type=str, default='Training',
                        help='train path(default: Training)')
    # parser.add_argument('--test_path', type=str, default='Test',
    #                     help='test path(default: Test)')
    parser.add_argument('--exposure_file_name', type=str, default='exposure.txt',
                        help='exposure file name(default: exposure.txt)')
    parser.add_argument('--ldr_folder_name', type=str, default=None,
                        help='ldr folder name(default: None)')
    parser.add_argument('--label_file_name', type=str, default='HDRImg.hdr',
                        help='label file name(default: HDRImg.hdr)')
    parser.add_argument('--mask_npy_name', type=str, default='masks.npy',
                        help='mask npy name(default: masks.npy)')
    parser.add_argument("--dataset_mode", type=str, default='train')
    parser.add_argument("--mask_mode", type=str, default='sam')
    parser.add_argument("--sam_checkpoint", type=str, default='./sam_vit_h_4b8939.pth')  # <<<===========================
    parser.add_argument("--model_type", type=str, default='default')
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=0,  # Requires open-cv to run post-processing
    )

    dataset_path = os.path.join(args.dataset_dir, args.train_path)  # <<<===========================
    for p in os.listdir(dataset_path):
        if args.ldr_folder_name == None:
            file_path = os.path.join(dataset_path, p)
        else:
            file_path = os.path.join(dataset_path, p, args.ldr_folder_name)
        if args.mask_mode == 'sam':
            sam_preprocess(file_path, mask_generator)
        elif args.mask_mode == 'random':
            random_preprocess(file_path)
        else:
            raise NotImplementedError
            
if __name__ == "__main__": 
    main()