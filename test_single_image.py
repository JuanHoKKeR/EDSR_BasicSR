#!/usr/bin/env python3
"""
Script simple para probar el modelo EDSR entrenado con una imagen individual
Uso: python test_single_image.py --image path/to/image.jpg --model path/to/model.pth
"""

import argparse
import cv2
import os
import torch
import numpy as np

from basicsr.archs.edsr_arch import EDSR
from basicsr.utils.img_util import img2tensor, tensor2img


def load_edsr_model(model_path, device):
    """Carga el modelo EDSR entrenado"""
    # Configuración basada en tu archivo YAML
    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=256,
        num_block=32,
        upscale=2,
        res_scale=0.1,
        img_range=255.,
        rgb_mean=[0.5, 0.5, 0.5]
    )
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Manejar diferentes formatos de checkpoint
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
        print("Usando parámetros EMA")
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
        print("Usando parámetros regulares")
    else:
        state_dict = checkpoint
        print("Usando state_dict directamente")
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    
    return model


def enhance_image(model, image_path, device):
    """Mejora una imagen usando el modelo EDSR"""
    # Leer imagen
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    print(f"Imagen original: {img.shape[1]}x{img.shape[0]} píxeles")
    
    # Convertir a tensor
    img_tensor = img2tensor(img, bgr2rgb=True, float32=True)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convertir de vuelta a imagen
    enhanced_img = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
    
    print(f"Imagen mejorada: {enhanced_img.shape[1]}x{enhanced_img.shape[0]} píxeles")
    
    return enhanced_img


def main():
    parser = argparse.ArgumentParser(description='Probar modelo EDSR con una imagen individual')
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Ruta a la imagen de entrada'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='experiments/EDSR_Microscopy_256to512/models/net_g_latest.pth',
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Ruta de salida (opcional, por defecto agrega "_enhanced" al nombre)'
    )
    
    args = parser.parse_args()
    
    # Verificar que existen los archivos
    if not os.path.exists(args.image):
        print(f"Error: No se encuentra la imagen {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: No se encuentra el modelo {args.model}")
        print("Modelos disponibles en experiments/EDSR_Microscopy_256to512/models/:")
        model_dir = "experiments/EDSR_Microscopy_256to512/models/"
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        return
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    try:
        # Cargar modelo
        print("Cargando modelo...")
        model = load_edsr_model(args.model, device)
        print("Modelo cargado exitosamente")
        
        # Procesar imagen
        print("Procesando imagen...")
        enhanced_img = enhance_image(model, args.image, device)
        
        # Determinar nombre de salida
        if args.output is None:
            base_name = os.path.splitext(args.image)[0]
            ext = os.path.splitext(args.image)[1]
            output_path = f"{base_name}_enhanced_x2{ext}"
        else:
            output_path = args.output
        
        # Guardar resultado
        cv2.imwrite(output_path, enhanced_img)
        print(f"Imagen mejorada guardada en: {output_path}")
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        return


if __name__ == '__main__':
    main()
