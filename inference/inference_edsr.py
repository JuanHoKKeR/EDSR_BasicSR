import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.edsr_arch import EDSR
from basicsr.utils.img_util import img2tensor, tensor2img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/EDSR_Microscopy_256to512/models/net_g_latest.pth',
        help='Path to the trained EDSR model'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='input_images', 
        help='Input image file or folder'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/EDSR_inference', 
        help='Output folder'
    )
    parser.add_argument(
        '--scale', 
        type=int, 
        default=2, 
        help='Upscaling factor'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='EDSR',
        help='Suffix for output images'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up EDSR model based on your training config
    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=256,  # From your config
        num_block=32,  # From your config
        upscale=args.scale,
        res_scale=0.1,
        img_range=255.,
        rgb_mean=[0.5, 0.5, 0.5]  # From your config
    )
    
    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
        print("Loaded EMA parameters")
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
        print("Loaded regular parameters")
    else:
        # Assume the checkpoint is the state_dict directly
        model.load_state_dict(checkpoint, strict=True)
        print("Loaded state dict directly")
    
    model.eval()
    model = model.to(device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        img_list = [args.input]
    elif os.path.isdir(args.input):
        # Folder of images
        img_list = sorted(glob.glob(os.path.join(args.input, '*.[jp][pn]g')))
        img_list.extend(sorted(glob.glob(os.path.join(args.input, '*.[JP][PN]G'))))
    else:
        raise ValueError(f"Input path {args.input} is neither a file nor a directory")
    
    if not img_list:
        print(f"No images found in {args.input}")
        return
    
    print(f"Found {len(img_list)} images to process")
    
    # Process each image
    for idx, img_path in enumerate(img_list):
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        print(f'Processing {idx+1}/{len(img_list)}: {imgname}')
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
            
        # Convert to tensor
        img_tensor = img2tensor(img, bgr2rgb=True, float32=True)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        try:
            with torch.no_grad():
                output = model(img_tensor)
        except Exception as error:
            print(f'Error processing {imgname}: {error}')
            continue
        
        # Convert back to image
        output_img = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
        
        # Save image
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}_x{args.scale}.png')
        cv2.imwrite(save_path, output_img)
        print(f'Saved: {save_path}')
    
    print(f'All images processed. Results saved in {args.output}')


if __name__ == '__main__':
    main()
