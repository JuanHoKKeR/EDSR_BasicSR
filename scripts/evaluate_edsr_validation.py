#!/usr/bin/env python3
"""
Evaluador Ultra-Eficiente de Memoria para EDSR
Para casos donde el evaluador normal falla por memoria limitada
Procesa imágenes en micro-lotes y usa técnicas agresivas de ahorro de memoria
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import gc
import psutil
import time
from pathlib import Path
from tqdm import tqdm
import warnings
import cv2
import sys

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports para utilidades de imagen BasicSR
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

warnings.filterwarnings('ignore')

class MemoryEfficientEDSREvaluator:
    """Evaluador ultra-eficiente de memoria para EDSR"""
    
    def __init__(self, model_path, max_memory_mb=8000):
        """
        Inicializa el evaluador ultra-eficiente
        
        Args:
            model_path: Ruta al modelo .pt optimizado
            max_memory_mb: Memoria máxima a usar en MB
        """
        self.model_path = model_path
        self.max_memory_mb = max_memory_mb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scale_factor = None
        
        # Configurar límites de memoria
        self._setup_memory_limits()
        
        # Cargar modelo
        self._load_model()
    
    def _setup_memory_limits(self):
        """Configura límites de memoria"""
        print(f"🧠 Configurando límites de memoria: {self.max_memory_mb} MB")
        
        # Obtener memoria disponible
        memory_info = psutil.virtual_memory()
        available_mb = memory_info.available / (1024**2)
        
        print(f"   Memoria total: {memory_info.total / (1024**2):.0f} MB")
        print(f"   Memoria disponible: {available_mb:.0f} MB")
        print(f"   Memoria configurada: {self.max_memory_mb} MB")
        
        if self.max_memory_mb > available_mb * 0.8:
            print(f"⚠️  Advertencia: Límite de memoria muy alto")
    
    def _load_model(self):
        """Carga el modelo optimizado"""
        print(f"📦 Cargando modelo ultra-eficiente: {self.model_path}")
        
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # Detectar factor de escala
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
            
            self.scale_factor = test_output.shape[-1] // test_input.shape[-1]
            
            del test_input, test_output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"✅ Modelo cargado - Escala: {self.scale_factor}x")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def _get_memory_usage(self):
        """Obtiene uso actual de memoria"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)
        
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
        
        return memory_mb, gpu_memory_mb
    
    def _aggressive_cleanup(self):
        """Limpieza agresiva de memoria"""
        # Limpiar Python garbage collector
        gc.collect()
        
        # Limpiar memoria PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _process_image_minimal(self, lr_path, hr_path):
        """Procesa una imagen con uso mínimo de memoria"""
        try:
            # Cargar imagen LR
            lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
            if lr_img is None:
                return None
            
            # Convertir a tensor
            lr_tensor = img2tensor(lr_img, bgr2rgb=True, float32=True)
            lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
            
            del lr_img  # Liberar inmediatamente
            
            # Inferencia
            with torch.no_grad():
                generated_batch = self.model(lr_tensor)
                generated = generated_batch.squeeze(0)
            
            del lr_tensor, generated_batch
            self._aggressive_cleanup()
            
            # Cargar imagen HR
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            if hr_img is None:
                del generated
                return None
            
            hr_tensor = img2tensor(hr_img, bgr2rgb=True, float32=True)
            hr_tensor = hr_tensor.to(self.device)
            
            del hr_img  # Liberar inmediatamente
            
            # Redimensionar HR si es necesario
            if generated.shape != hr_tensor.shape:
                hr_tensor = torch.nn.functional.interpolate(
                    hr_tensor.unsqueeze(0), 
                    size=generated.shape[-2:], 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze(0)
            
            # Calcular métricas básicas (solo PSNR y SSIM para ahorrar memoria)
            generated_np = tensor2img(generated.unsqueeze(0), rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
            hr_np = tensor2img(hr_tensor.unsqueeze(0), rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
            
            psnr = calculate_psnr(generated_np, hr_np, crop_border=0, test_y_channel=False)
            ssim = calculate_ssim(generated_np, hr_np, crop_border=0, test_y_channel=False)
            
            # Limpiar todo
            del generated, hr_tensor, generated_np, hr_np
            self._aggressive_cleanup()
            
            return {
                'psnr': float(psnr),
                'ssim': float(ssim),
                'lr_path': lr_path,
                'hr_path': hr_path
            }
            
        except Exception as e:
            print(f"❌ Error en imagen {lr_path}: {e}")
            self._aggressive_cleanup()
            return None
    
    def evaluate_dataset(self, lr_meta_file, hr_meta_file, output_dir, model_name, 
                        base_path="", max_images=None, checkpoint_interval=25):
        """
        Evalúa dataset completo con uso ultra-eficiente de memoria
        
        Args:
            lr_meta_file: Archivo meta LR
            hr_meta_file: Archivo meta HR
            output_dir: Directorio de salida
            model_name: Nombre del modelo
            base_path: Ruta base
            max_images: Máximo número de imágenes
            checkpoint_interval: Intervalo para guardar checkpoints
        """
        print(f"\n🚀 Evaluación ultra-eficiente: {model_name}")
        print(f"💾 Memoria máxima configurada: {self.max_memory_mb} MB")
        
        # Cargar rutas
        with open(lr_meta_file, 'r') as f:
            lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        with open(hr_meta_file, 'r') as f:
            hr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        image_pairs = list(zip(lr_paths, hr_paths))
        
        if max_images:
            image_pairs = image_pairs[:max_images]
        
        print(f"📊 Procesando {len(image_pairs)} pares de imágenes")
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Archivo de checkpoint
        checkpoint_file = os.path.join(output_dir, f"{model_name}_checkpoint.csv")
        
        # Cargar progreso previo si existe
        start_index = 0
        results = []
        
        if os.path.exists(checkpoint_file):
            try:
                prev_results = pd.read_csv(checkpoint_file)
                results = prev_results.to_dict('records')
                start_index = len(results)
                print(f"📋 Reanudando desde imagen {start_index}")
            except:
                print("⚠️  No se pudo cargar checkpoint previo, iniciando desde cero")
        
        # Procesar imágenes
        successful = 0
        failed = 0
        
        for i in tqdm(range(start_index, len(image_pairs)), desc="Procesando"):
            lr_path, hr_path = image_pairs[i]
            
            # Monitorear memoria antes del procesamiento
            ram_mb, gpu_mb = self._get_memory_usage()
            
            if ram_mb > self.max_memory_mb * 0.9:
                print(f"⚠️  Memoria alta ({ram_mb:.0f} MB), limpiando agresivamente...")
                self._aggressive_cleanup()
                time.sleep(1)  # Dar tiempo para la limpieza
            
            # Procesar imagen
            result = self._process_image_minimal(lr_path, hr_path)
            
            if result:
                result['image_index'] = i + 1
                result['lr_filename'] = os.path.basename(lr_path)
                result['hr_filename'] = os.path.basename(hr_path)
                results.append(result)
                successful += 1
            else:
                failed += 1
            
            # Guardar checkpoint periódicamente
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(results, checkpoint_file)
                ram_mb, gpu_mb = self._get_memory_usage()
                print(f"💾 Checkpoint: {i+1}/{len(image_pairs)} - RAM: {ram_mb:.0f}MB, GPU: {gpu_mb:.0f}MB")
        
        # Guardar resultados finales
        if results:
            df = pd.DataFrame(results)
            final_file = os.path.join(output_dir, f"{model_name}_metrics_memory_efficient.csv")
            df.to_csv(final_file, index=False)
            
            # Remover checkpoint temporal
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            
            # Estadísticas
            print(f"\n📈 Evaluación completada:")
            print(f"   ✅ Exitosas: {successful}")
            print(f"   ❌ Fallidas: {failed}")
            print(f"   📊 PSNR promedio: {df['psnr'].mean():.2f} dB")
            print(f"   📊 SSIM promedio: {df['ssim'].mean():.4f}")
            print(f"💾 Resultados: {final_file}")
        
        # Limpieza final
        self._aggressive_cleanup()
    
    def _save_checkpoint(self, results, checkpoint_file):
        """Guarda checkpoint de progreso"""
        if results:
            df = pd.DataFrame(results)
            df.to_csv(checkpoint_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluador Ultra-Eficiente de Memoria para EDSR")
    
    parser.add_argument("--model_path", required=True, help="Ruta al modelo .pt")
    parser.add_argument("--model_name", required=True, help="Nombre del modelo")
    parser.add_argument("--lr_meta_file", required=True, help="Archivo meta LR")
    parser.add_argument("--hr_meta_file", required=True, help="Archivo meta HR")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Directorio de salida")
    parser.add_argument("--base_path", default="", help="Ruta base")
    parser.add_argument("--max_images", type=int, help="Máximo número de imágenes")
    parser.add_argument("--max_memory_mb", type=int, default=6000, help="Memoria máxima en MB")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Intervalo de checkpoint")
    
    args = parser.parse_args()
    
    print("🧠 EVALUADOR ULTRA-EFICIENTE DE MEMORIA PARA EDSR")
    print("=" * 60)
    print("⚠️  NOTA: Este evaluador solo calcula PSNR y SSIM para ahorrar memoria")
    print("   Para evaluación completa con KimiaNet, usa el evaluador optimizado normal")
    print("=" * 60)
    
    try:
        evaluator = MemoryEfficientEDSREvaluator(args.model_path, args.max_memory_mb)
        
        evaluator.evaluate_dataset(
            lr_meta_file=args.lr_meta_file,
            hr_meta_file=args.hr_meta_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            base_path=args.base_path,
            max_images=args.max_images,
            checkpoint_interval=args.checkpoint_interval
        )
        
        print("\n🎉 Evaluación ultra-eficiente completada!")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())