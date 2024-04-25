import numpy as np
import torch
from torch.nn import functional as F

import os
import re
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor




def get_arguments():
    
    parser = argparse.ArgumentParser()

    #parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    #parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    # images_path = args.data + '/Images/'
    # masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    # for obj_name in os.listdir(images_path):
    #     if ".DS" not in obj_name:
    
    persam(args, output_path)


def persam(args, output_path):

    print("\n------------> SAM is Segmenting Dogs " )
    
    # Path preparation
    # ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    # ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    # test_images_path = os.path.join(images_path, obj_name)
    
    masks_path = "./training_masks"

    def extract_numeric_part(filename):
    # Extract the numeric part from the filename
        match = re.search(r"\((\d+)\)", filename)
        if match:
            return int(match.group(1))
        return float("inf")  # If no numeric part found, put it at the end

    path = masks_path  # Replace with your actual path

    # Get a list of filenames in the directory
    filenames = os.listdir(path)

    # Sort filenames based on the numeric part
    sorted_filenames = sorted(filenames, key=extract_numeric_part)

    prompt_label_masks = []

    # Print the sorted filenames
    for img in sorted_filenames:
        
        mask_img = cv2.imread(os.path.join(masks_path,img))
        if mask_img is not None:
            prompt_label_masks.append(mask_img)
            
    images_path = 'samples.npy'
            
    image_data = np.load(images_path,allow_pickle=True)
    

    output_path = os.path.join(output_path, 'SAM_OUTPUT_100_IMAGES')
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    
    print("======> Load SAM" )
    
    for i in range(300):
        ref_image = image_data[i]
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    
        ref_mask = prompt_label_masks[i]
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
    

        if args.sam_type == 'vit_h':
            sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
        elif args.sam_type == 'vit_t':
            sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
            sam.eval()

        predictor = SamPredictor(sam)

        print("Reading image ",i )
        # Image features encoding
        ref_mask = predictor.set_image(ref_image, ref_mask)
        ref_feat = predictor.features.squeeze().permute(1, 2, 0)

        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction
        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)


    print('======> Start Testing')
    for iteration in range(300,400):
        it_num = iteration 
        # Load test image
        # test_idx = '%02d' % test_idx
        # test_image_path = test_images_path + '/' + test_idx + '.jpg'
        iteration = '%02d' % iteration
        test_image = image_data[it_num]
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :], 
            multimask_output=True)
        best_idx = np.argmax(scores)

        # Save masks
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(masks[best_idx], plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{iteration}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

        final_mask = masks[best_idx]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, iteration +  '.png')
        cv2.imwrite(mask_output_path, mask_colors)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label
    

if __name__ == "__main__":
    main()
