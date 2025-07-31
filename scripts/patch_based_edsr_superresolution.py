#!/usr/bin/env python3
"""
Super-Resolución Basada en Parches EDSR
Divide imagen en parches, aplica modelo, reconstruye imagen completa
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,  # Cambia a True si tienes LaTeX instalado
    'font.size': 12
})
from PIL import Image
from pathlib import Path
import json
import warnings
import cv2
import time
warnings.filterwarnings('ignore')

# Importar pytorch-msssim si está disponible
try:
    from pytorch_msssim import ssim, ms_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    print("⚠️  pytorch-msssim no disponible. MS-SSIM no estará disponible.")
    MSSSIM_AVAILABLE = False

class PatchBasedSuperResolution:
    """Aplicación de super-resolución por parches usando EDSR"""
    
    def __init__(self, model_path, patch_size=64):
        """
        Inicializa el sistema de super-resolución por parches
        
        Args:
            model_path: Ruta al modelo EDSR (.pt)
            patch_size: Tamaño de los parches (debe coincidir con entrada del modelo)
        """
        self.patch_size = patch_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Carga el modelo EDSR"""
        print(f"📦 Cargando modelo desde: {model_path}")
        print(f"🔧 Usando dispositivo: {self.device}")
        
        try:
            # Cargar modelo optimizado (TorchScript)
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.eval()
            
            print("✅ Modelo EDSR cargado correctamente")
            
            # Verificar funcionamiento y determinar escala
            test_input = torch.randn(1, 3, self.patch_size, self.patch_size).to(self.device)
            
            with torch.no_grad():
                test_output = self.model(test_input)
            
            self.scale_factor = test_output.shape[2] // test_input.shape[2]
            self.output_patch_size = test_output.shape[2]
            
            print(f"🔍 Modelo detectado:")
            print(f"   Input: {test_input.shape[2]}x{test_input.shape[3]} → Output: {test_output.shape[2]}x{test_output.shape[3]}")
            print(f"   Factor de escala: ×{self.scale_factor}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def load_image(self, image_path):
        """Carga una imagen y la convierte a tensor PyTorch"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encuentra el archivo: {image_path}")
            
            # Cargar imagen con OpenCV (BGR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a float32 y normalizar a [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convertir a tensor PyTorch [H, W, C]
            image_tensor = torch.from_numpy(image)
            
            # Verificar que la imagen se cargó correctamente
            if len(image_tensor.shape) != 3 or image_tensor.shape[-1] != 3:
                raise ValueError(f"Imagen inválida. Esperado: [H,W,3], obtenido: {image_tensor.shape}")
                
            return image_tensor
        except Exception as e:
            print(f"❌ Error cargando imagen {image_path}: {e}")
            return None
    
    def extract_patches(self, image, patch_size):
        """
        Extrae parches de una imagen
        
        Args:
            image: Imagen tensor [H, W, C]
            patch_size: Tamaño del parche
            
        Returns:
            Lista de parches y información de grilla
        """
        h, w = image.shape[:2]
        
        # Calcular número de parches
        patches_h = h // patch_size
        patches_w = w // patch_size
        
        print(f"📐 Dividiendo imagen {h}x{w} en {patches_h}x{patches_w} parches de {patch_size}x{patch_size}")
        
        if h % patch_size != 0 or w % patch_size != 0:
            print(f"⚠️  Advertencia: La imagen no se divide exactamente en parches de {patch_size}x{patch_size}")
            print(f"    Se usarán solo los parches completos")
        
        patches = []
        patch_positions = []
        
        for i in range(patches_h):
            for j in range(patches_w):
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size
                
                patch = image[y_start:y_end, x_start:x_end, :]
                patches.append(patch)
                patch_positions.append((i, j, y_start, y_end, x_start, x_end))
        
        grid_info = {
            'patches_h': patches_h,
            'patches_w': patches_w,
            'patch_size': patch_size,
            'original_size': (h, w),
            'positions': patch_positions
        }
        
        print(f"✅ Extraídos {len(patches)} parches")
        return patches, grid_info
    
    def process_patches(self, patches):
        """
        Procesa cada parche con el modelo EDSR
        
        Args:
            patches: Lista de parches tensor [H, W, C]
            
        Returns:
            Lista de parches procesados (super-resolución)
        """
        processed_patches = []
        
        print(f"🚀 Procesando {len(patches)} parches con EDSR...")
        
        for i, patch in enumerate(patches):
            print(f"   Procesando parche {i+1}/{len(patches)}", end='\r')
            
            # Convertir a formato [1, C, H, W] para PyTorch
            patch_batch = patch.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Aplicar modelo
            with torch.no_grad():
                enhanced_batch = self.model(patch_batch)
                enhanced_batch = torch.clamp(enhanced_batch, 0, 1)
            
            # Convertir de vuelta a formato [H, W, C] y mover a CPU
            enhanced_patch = enhanced_batch.squeeze(0).permute(1, 2, 0).cpu()
            
            processed_patches.append(enhanced_patch)
        
        print(f"\n✅ Todos los parches procesados")
        return processed_patches
    
    def reconstruct_image(self, processed_patches, grid_info):
        """
        Reconstruye la imagen completa a partir de parches procesados
        
        Args:
            processed_patches: Lista de parches super-resueltos
            grid_info: Información de la grilla original
            
        Returns:
            Imagen reconstruida
        """
        patches_h = grid_info['patches_h']
        patches_w = grid_info['patches_w']
        original_h, original_w = grid_info['original_size']
        
        # Calcular tamaño de la imagen reconstruida
        reconstructed_h = original_h * self.scale_factor
        reconstructed_w = original_w * self.scale_factor
        
        print(f"🔧 Reconstruyendo imagen: {original_h}x{original_w} → {reconstructed_h}x{reconstructed_w}")
        
        # Inicializar imagen reconstruida
        reconstructed = torch.zeros(reconstructed_h, reconstructed_w, 3)
        
        # Colocar cada parche en su posición
        patch_idx = 0
        for i in range(patches_h):
            for j in range(patches_w):
                # Calcular posición en la imagen reconstruida
                y_start = i * self.output_patch_size
                y_end = y_start + self.output_patch_size
                x_start = j * self.output_patch_size
                x_end = x_start + self.output_patch_size
                
                # Colocar parche
                reconstructed[y_start:y_end, x_start:x_end, :] = processed_patches[patch_idx]
                
                patch_idx += 1
    
        print(f"✅ Imagen reconstruida: {reconstructed.shape}")
        return reconstructed
    
    def calculate_metrics(self, generated, ground_truth):
        """Calcula métricas de evaluación"""
        # Asegurar mismo tamaño
        if generated.shape != ground_truth.shape:
            print(f"⚠️  Redimensionando ground truth: {ground_truth.shape} → {generated.shape}")
            ground_truth = F.interpolate(
                ground_truth.unsqueeze(0).permute(2, 0, 1).unsqueeze(0), 
                size=generated.shape[:2], 
                mode='bicubic'
            ).squeeze(0).permute(1, 2, 0)
        
        # Calcular PSNR
        mse = torch.mean((generated - ground_truth) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # Calcular SSIM y MS-SSIM usando pytorch-msssim
        ssim_val = None
        ms_ssim_val = None
        
        if MSSSIM_AVAILABLE:
            try:
                # Convertir a formato [1, C, H, W]
                gen_tensor = generated.permute(2, 0, 1).unsqueeze(0)
                gt_tensor = ground_truth.permute(2, 0, 1).unsqueeze(0)
                
                ssim_val = ssim(gen_tensor, gt_tensor, data_range=1.0, size_average=True)
                ms_ssim_val = ms_ssim(gen_tensor, gt_tensor, data_range=1.0, size_average=True)
                
                ssim_val = float(ssim_val)
                ms_ssim_val = float(ms_ssim_val)
            except Exception as e:
                print(f"⚠️  Error calculando SSIM: {e}")
        
        return {
            'psnr': float(psnr),
            'ssim': ssim_val,
            'ms_ssim': ms_ssim_val,
            'mse': float(mse)
        }
    
    def calculate_absolute_difference(self, generated, ground_truth):
        """Calcula diferencia absoluta"""
        # Diferencia absoluta
        abs_diff = torch.abs(generated - ground_truth)
        
        return abs_diff
    
    def save_results(self, generated_image, ground_truth, abs_diff, metrics, output_dir, image_name):
        """Guarda todos los resultados"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar imagen generada
        generated_path = os.path.join(output_dir, f"{image_name}_patch_reconstructed.png")
        generated_pil = Image.fromarray((torch.clamp(generated_image, 0, 1) * 255).numpy().astype(np.uint8))
        generated_pil.save(generated_path)
        print(f"💾 Imagen reconstruida guardada: {generated_path}")

        # Guardar diferencia absoluta
        diff_path = os.path.join(output_dir, f"{image_name}_absolute_difference.png")

        # Crear visualización de diferencia
        plt.figure(figsize=(12, 5))
        
        # Imagen reconstruida
        plt.subplot(1, 3, 1)
        plt.imshow(generated_pil)
        plt.title('Reconstrucción por Parches', fontweight='bold')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        gt_pil = Image.fromarray((torch.clamp(ground_truth, 0, 1) * 255).numpy().astype(np.uint8))
        plt.imshow(gt_pil)
        plt.title('Imagen de Referencia', fontweight='bold')
        plt.axis('off')
        
        # Diferencia absoluta
        plt.subplot(1, 3, 3)
        diff_display = torch.mean(abs_diff, dim=-1).numpy()  # Promedio de canales para visualización
        im = plt.imshow(diff_display, cmap='hot', vmin=0, vmax=0.3)
        plt.title(f'Diferencia Absoluta\nError Promedio: {np.mean(diff_display):.4f}', fontweight='bold')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Título general con métricas
        plt.suptitle(f'Resultados de Super-Resolución por Parches - {image_name}\n' +
                    f'PSNR: {metrics["psnr"]:.4f} dB | SSIM: {metrics.get("ssim", "N/A")} | MSE: {metrics["mse"]:.6f}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(diff_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Análisis de diferencia guardado: {diff_path}")

        # Guardar métricas en JSON
        metrics_path = os.path.join(output_dir, f"{image_name}_patch_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'patch_info': {
                    'model_scale_factor': self.scale_factor,
                    'patch_size': self.patch_size,
                    'output_patch_size': self.output_patch_size,
                    'generated_size': list(generated_image.shape[:2]),
                    'ground_truth_size': list(ground_truth.shape[:2])
                }
            }, f, indent=2)
        print(f"📄 Métricas guardadas: {metrics_path}")

    def process_image_pair(self, input_lr_path, ground_truth_hr_path, output_dir, image_name=None):
        """
        Procesa un par de imágenes: LR → parches → reconstrucción HR
        
        Args:
            input_lr_path: Imagen de entrada de baja resolución
            ground_truth_hr_path: Imagen ground truth de alta resolución
            output_dir: Directorio de salida
            image_name: Nombre para archivos de salida
        """
        if image_name is None:
            image_name = Path(input_lr_path).stem

        print(f"\n🎯 PROCESANDO: {image_name}")
        print("=" * 50)
        
        # Cargar imágenes
        print("📂 Cargando imágenes...")
        input_lr = self.load_image(input_lr_path)
        ground_truth_hr = self.load_image(ground_truth_hr_path)
        
        if input_lr is None or ground_truth_hr is None:
            print("❌ Error cargando imágenes")
            return None

        print(f"   Input LR: {input_lr.shape}")
        print(f"   Ground Truth HR: {ground_truth_hr.shape}")

        # Verificar si necesita redimensionamiento de entrada
        expected_input_size = ground_truth_hr.shape[0] // self.scale_factor
        if input_lr.shape[0] != expected_input_size or input_lr.shape[1] != expected_input_size:
            print(f"⚠️  Advertencia: Redimensionando imagen de entrada a {expected_input_size}x{expected_input_size}...")
            input_lr = F.interpolate(
                input_lr.unsqueeze(0).permute(2, 0, 1).unsqueeze(0), 
                size=[expected_input_size, expected_input_size], 
                mode='bicubic'
            ).squeeze(0).permute(1, 2, 0)
        
        # Extraer parches
        patches, grid_info = self.extract_patches(input_lr, self.patch_size)
        
        # Procesar parches
        processed_patches = self.process_patches(patches)
        
        # Reconstruir imagen
        reconstructed_hr = self.reconstruct_image(processed_patches, grid_info)
        
        # Calcular métricas
        print("📊 Calculando métricas...")
        metrics = self.calculate_metrics(reconstructed_hr, ground_truth_hr)
        
        # Calcular diferencia absoluta
        abs_diff = self.calculate_absolute_difference(reconstructed_hr, ground_truth_hr)
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS:")
        print(f"   PSNR: {metrics['psnr']:.4f} dB")
        if metrics['ssim'] is not None:
            print(f"   SSIM: {metrics['ssim']:.4f}")
        if metrics['ms_ssim'] is not None:
            print(f"   MS-SSIM: {metrics['ms_ssim']:.4f}")
        print(f"   MSE: {metrics['mse']:.6f}")

        # Guardar resultados
        self.save_results(reconstructed_hr, ground_truth_hr, abs_diff, metrics, output_dir, image_name)

        print(f"✅ Procesamiento completado para {image_name}")
        return metrics

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Super-Resolución Basada en Parches EDSR")
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo EDSR (.pt)"
    )
    
    parser.add_argument(
        "--input_lr",
        required=True,
        help="Imagen de entrada de baja resolución"
    )
    
    parser.add_argument(
        "--ground_truth_hr",
        required=True,
        help="Imagen ground truth de alta resolución"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./patch_based_results",
        help="Directorio para guardar resultados"
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Tamaño de parches (debe coincidir con entrada del modelo)"
    )
    
    parser.add_argument(
        "--image_name",
        default=None,
        help="Nombre personalizado para archivos de salida"
    )
    
    args = parser.parse_args()
    
    # Verificar archivos
    for file_path in [args.model_path, args.input_lr, args.ground_truth_hr]:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra: {file_path}")
            return 1

    print("🧩 SUPER-RESOLUCIÓN BASADA EN PARCHES - EDSR")
    print("=" * 50)
    print(f"Modelo: {args.model_path}")
    print(f"Input LR: {args.input_lr}")
    print(f"Ground Truth HR: {args.ground_truth_hr}")
    print(f"Tamaño de parche: {args.patch_size}x{args.patch_size}")
    print(f"Output: {args.output_dir}")

    try:
        # Crear procesador
        processor = PatchBasedSuperResolution(args.model_path, args.patch_size)
        
        # Procesar imágenes
        metrics = processor.process_image_pair(
            args.input_lr,
            args.ground_truth_hr,
            args.output_dir,
            args.image_name
        )
        
        if metrics:
            print(f"\n🎉 Procesamiento exitoso!")
            print(f"📂 Resultados guardados en: {args.output_dir}")

        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())