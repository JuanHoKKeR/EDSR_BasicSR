#!/usr/bin/env python3
"""
Inferencia EDSR con Modelos Optimizados (.pt)
Procesa imágenes individuales o carpetas usando modelos TorchScript optimizados
Compatible con modelos exportados desde BasicSR
"""

import argparse
import cv2
import numpy as np
import os
import sys
import torch
import glob
from pathlib import Path
from collections import OrderedDict

# Agregar el directorio raíz al path para importar BasicSR
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

class EDSROptimizedInference:
    """Clase para inferencia con modelos EDSR optimizados"""
    
    def __init__(self, model_path, device='auto'):
        """
        Inicializa el inferenciador EDSR
        
        Args:
            model_path: Ruta al modelo .pt optimizado
            device: 'cpu', 'cuda' o 'auto'
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = self._load_optimized_model()
        
    def _setup_device(self, device):
        """Configura el dispositivo"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"🚀 Usando dispositivo: {device}")
        return device
    
    def _load_optimized_model(self):
        """Carga el modelo optimizado .pt"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")
        
        print(f"📦 Cargando modelo optimizado desde: {self.model_path}")
        
        try:
            # Cargar modelo TorchScript optimizado
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            
            print("✅ Modelo EDSR optimizado cargado correctamente")
            
            # Verificar que funciona con una imagen de prueba
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                test_output = model(test_input)
            
            # Inferir factor de escala automáticamente
            scale_factor = test_output.shape[-1] // test_input.shape[-1]
            print(f"🔍 Factor de escala detectado: {scale_factor}x")
            print(f"🧪 Prueba exitosa - Input: {test_input.shape}, Output: {test_output.shape}")
            
            self.scale_factor = scale_factor
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo optimizado: {e}")
            raise
    
    def preprocess_image(self, img_path):
        """
        Preprocesa imagen para EDSR exactamente como tu test_single_image.py
        
        Args:
            img_path: Ruta a la imagen
            
        Returns:
            tensor: Imagen procesada como tensor
            original_shape: Forma original (H, W)
        """
        # Cargar imagen
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        
        # Guardar forma original
        original_shape = img.shape[:2]  # (H, W)
        
        # Convertir a tensor usando BasicSR (exactamente como tu código)
        img_tensor = img2tensor(img, bgr2rgb=True, float32=True)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Agregar batch dimension
        
        return img_tensor, original_shape
    
    def postprocess_output(self, output_tensor):
        """
        Postprocesa la salida del modelo exactamente como tu test_single_image.py
        
        Args:
            output_tensor: Tensor de salida del modelo
            
        Returns:
            numpy array: Imagen procesada en formato BGR uint8
        """
        # Convertir tensor a imagen usando BasicSR exactamente como tu código
        # IMPORTANTE: usar min_max=(0, 255) no (0, 1) porque img_range=255
        enhanced_img = tensor2img(output_tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
        
        return enhanced_img
    
    def process_single_image(self, input_path, output_path=None, gt_path=None):
        """
        Procesa una imagen individual
        
        Args:
            input_path: Ruta de la imagen de entrada
            output_path: Ruta de salida (opcional)
            gt_path: Ruta del ground truth (opcional, para métricas)
            
        Returns:
            dict: Resultados del procesamiento
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No se encontró la imagen: {input_path}")
        
        # Determinar ruta de salida
        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(input_dir, f"{input_name}_EDSR_x{self.scale_factor}.png")
        
        print(f"🖼️  Procesando: {os.path.basename(input_path)}")
        
        # Preprocesar imagen
        input_tensor, original_shape = self.preprocess_image(input_path)
        
        # Inferencia
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Postprocesar salida
        output_img = self.postprocess_output(output_tensor)
        
        # Guardar imagen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img)
        print(f"💾 Resultado guardado en: {output_path}")
        
        # Calcular métricas si hay ground truth
        metrics = {}
        if gt_path and os.path.exists(gt_path):
            metrics = self._calculate_metrics(output_img, gt_path)
            self._print_metrics(metrics)
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'gt_path': gt_path,
            'metrics': metrics,
            'input_shape': original_shape,
            'output_shape': output_img.shape[:2]
        }
    
    def process_folder(self, input_folder, output_folder=None, gt_folder=None, extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')):
        """
        Procesa una carpeta completa de imágenes
        
        Args:
            input_folder: Carpeta de imágenes de entrada
            output_folder: Carpeta de salida (opcional)
            gt_folder: Carpeta de ground truth (opcional)
            extensions: Extensiones de archivo a procesar
            
        Returns:
            dict: Resultados del procesamiento
        """
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"No se encontró la carpeta: {input_folder}")
        
        # Configurar carpeta de salida
        if output_folder is None:
            output_folder = os.path.join(os.path.dirname(input_folder.rstrip('/')), 
                                       f"{os.path.basename(input_folder.rstrip('/'))}_EDSR_x{self.scale_factor}")
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Procesando carpeta: {input_folder}")
        print(f"📂 Guardando en: {output_folder}")
        
        # Encontrar todas las imágenes
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        image_paths = sorted(image_paths)
        print(f"🔍 Encontradas {len(image_paths)} imágenes")
        
        if len(image_paths) == 0:
            print("⚠️  No se encontraron imágenes en la carpeta")
            return {}
        
        # Procesar cada imagen
        results = []
        all_metrics = OrderedDict()
        all_metrics['psnr'] = []
        all_metrics['ssim'] = []
        
        for i, img_path in enumerate(image_paths, 1):
            try:
                # Determinar rutas
                img_name = os.path.basename(img_path)
                output_path = os.path.join(output_folder, img_name)
                
                gt_path = None
                if gt_folder and os.path.exists(gt_folder):
                    gt_path = os.path.join(gt_folder, img_name)
                    if not os.path.exists(gt_path):
                        gt_path = None
                
                print(f"\n[{i}/{len(image_paths)}] {img_name}")
                
                # Procesar imagen
                result = self.process_single_image(img_path, output_path, gt_path)
                results.append(result)
                
                # Acumular métricas
                if result['metrics']:
                    all_metrics['psnr'].append(result['metrics']['psnr'])
                    all_metrics['ssim'].append(result['metrics']['ssim'])
                
            except Exception as e:
                print(f"❌ Error procesando {img_name}: {e}")
                continue
        
        # Calcular métricas promedio
        summary_metrics = {}
        if all_metrics['psnr']:
            summary_metrics = {
                'avg_psnr': np.mean(all_metrics['psnr']),
                'avg_ssim': np.mean(all_metrics['ssim']),
                'std_psnr': np.std(all_metrics['psnr']),
                'std_ssim': np.std(all_metrics['ssim']),
                'processed_images': len(results)
            }
            
            print(f"\n📊 RESUMEN FINAL:")
            print(f"   Imágenes procesadas: {summary_metrics['processed_images']}")
            print(f"   PSNR promedio: {summary_metrics['avg_psnr']:.2f} ± {summary_metrics['std_psnr']:.2f} dB")
            print(f"   SSIM promedio: {summary_metrics['avg_ssim']:.4f} ± {summary_metrics['std_ssim']:.4f}")
        
        return {
            'input_folder': input_folder,
            'output_folder': output_folder,
            'results': results,
            'summary_metrics': summary_metrics
        }
    
    def _calculate_metrics(self, output_img, gt_path):
        """Calcula métricas PSNR y SSIM"""
        try:
            # Cargar ground truth
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            if gt_img is None:
                return {}
            
            # Asegurar que ambas imágenes tengan el mismo tamaño
            if output_img.shape != gt_img.shape:
                # Redimensionar GT al tamaño de la salida
                gt_img = cv2.resize(gt_img, (output_img.shape[1], output_img.shape[0]), 
                                  interpolation=cv2.INTER_CUBIC)
            
            # Calcular métricas usando BasicSR
            psnr = calculate_psnr(output_img, gt_img, crop_border=0, test_y_channel=False)
            ssim = calculate_ssim(output_img, gt_img, crop_border=0, test_y_channel=False)
            
            return {
                'psnr': float(psnr),
                'ssim': float(ssim)
            }
            
        except Exception as e:
            print(f"⚠️  Error calculando métricas: {e}")
            return {}
    
    def _print_metrics(self, metrics):
        """Imprime métricas"""
        if metrics:
            print(f"📊 Métricas:")
            print(f"   PSNR: {metrics['psnr']:.2f} dB")
            print(f"   SSIM: {metrics['ssim']:.4f}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Inferencia EDSR con Modelos Optimizados (.pt)")
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo EDSR optimizado (.pt)"
    )
    
    # Opción para imagen individual
    parser.add_argument(
        "--input_path",
        help="Ruta a imagen de entrada individual"
    )
    
    parser.add_argument(
        "--output_path",
        help="Ruta de salida para imagen individual (opcional)"
    )
    
    parser.add_argument(
        "--gt_path",
        help="Ruta al ground truth para métricas (opcional)"
    )
    
    # Opción para carpeta
    parser.add_argument(
        "--input_folder",
        help="Carpeta de imágenes de entrada"
    )
    
    parser.add_argument(
        "--output_folder",
        help="Carpeta de salida (opcional)"
    )
    
    parser.add_argument(
        "--gt_folder",
        help="Carpeta de ground truth para métricas (opcional)"
    )
    
    # Configuración
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help="Dispositivo a usar"
    )
    
    parser.add_argument(
        "--extensions",
        nargs='+',
        default=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'],
        help="Extensiones de archivo a procesar"
    )
    
    args = parser.parse_args()
    
    # Verificar argumentos
    if not args.input_path and not args.input_folder:
        print("❌ Error: Especifica --input_path o --input_folder")
        return 1
    
    if args.input_path and args.input_folder:
        print("❌ Error: Especifica solo --input_path O --input_folder, no ambos")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: No se encontró el modelo: {args.model_path}")
        return 1
    
    print("🖼️  INFERENCIA EDSR CON MODELO OPTIMIZADO")
    print("=" * 50)
    print(f"Modelo: {args.model_path}")
    print(f"Dispositivo: {args.device}")
    
    try:
        # Inicializar inferenciador
        inferencer = EDSROptimizedInference(args.model_path, args.device)
        
        if args.input_path:
            # Procesar imagen individual
            print(f"Entrada: {args.input_path}")
            if args.output_path:
                print(f"Salida: {args.output_path}")
            if args.gt_path:
                print(f"Ground Truth: {args.gt_path}")
            
            result = inferencer.process_single_image(
                args.input_path, 
                args.output_path, 
                args.gt_path
            )
            
            print(f"\n🎉 Procesamiento completado exitosamente!")
            
        else:
            # Procesar carpeta
            print(f"Carpeta entrada: {args.input_folder}")
            if args.output_folder:
                print(f"Carpeta salida: {args.output_folder}")
            if args.gt_folder:
                print(f"Carpeta GT: {args.gt_folder}")
            
            result = inferencer.process_folder(
                args.input_folder,
                args.output_folder,
                args.gt_folder,
                args.extensions
            )
            
            print(f"\n🎉 Procesamiento de carpeta completado!")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante la inferencia: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())