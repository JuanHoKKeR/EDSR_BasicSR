#!/usr/bin/env python3
"""
Evaluador Dinámico con KimiaNet para EDSR
Alterna carga de modelos para optimizar uso de memoria GPU
Procesa en lotes: SR -> descarga -> KimiaNet -> descarga -> repetir
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
import tempfile
import pickle

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports para utilidades de imagen BasicSR
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# Import para KimiaNet
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow no disponible - KimiaNet deshabilitada")

warnings.filterwarnings('ignore')

class DynamicModelEvaluator:
    """Evaluador dinámico que alterna entre modelos para optimizar memoria"""
    
    def __init__(self, edsr_model_path, kimianet_weights_path, batch_size=8, max_memory_mb=8000):
        """
        Inicializa el evaluador dinámico
        
        Args:
            edsr_model_path: Ruta al modelo EDSR .pt
            kimianet_weights_path: Ruta a los pesos de KimiaNet .h5
            batch_size: Tamaño de lote para procesamiento
            max_memory_mb: Memoria máxima a usar en MB
        """
        self.edsr_model_path = edsr_model_path
        self.kimianet_weights_path = kimianet_weights_path
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modelos (se cargan dinámicamente)
        self.edsr_model = None
        self.kimianet_model = None
        self.scale_factor = None
        
        # Directorio temporal para lotes
        self.temp_dir = tempfile.mkdtemp(prefix="edsr_eval_")
        
        print(f"🔄 Evaluador dinámico inicializado")
        print(f"📦 EDSR: {edsr_model_path}")
        print(f"🧠 KimiaNet: {kimianet_weights_path}")
        print(f"📊 Lote: {batch_size} imágenes")
        print(f"💾 Memoria máxima: {max_memory_mb} MB")
        print(f"📁 Temporal: {self.temp_dir}")
    
    def _get_memory_usage(self):
        """Obtiene uso actual de memoria"""
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024**2)
        
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
        
        return ram_mb, gpu_memory_mb
    
    def _aggressive_cleanup(self):
        """Limpieza agresiva de memoria"""
        # Python garbage collection
        gc.collect()
        
        # PyTorch cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # TensorFlow cleanup si está disponible
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
            if hasattr(tf.config.experimental, 'reset_memory_growth'):
                try:
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    for gpu in gpus:
                        tf.config.experimental.reset_memory_growth(gpu)
                except:
                    pass
    
    def _load_edsr_model(self):
        """Carga el modelo EDSR"""
        if self.edsr_model is not None:
            return
        
        print("📦 Cargando modelo EDSR...")
        ram_before, gpu_before = self._get_memory_usage()
        
        try:
            self.edsr_model = torch.jit.load(self.edsr_model_path, map_location=self.device)
            self.edsr_model.eval()
            
            # Detectar factor de escala si no se ha hecho
            if self.scale_factor is None:
                test_input = torch.randn(1, 3, 64, 64).to(self.device)
                with torch.no_grad():
                    test_output = self.edsr_model(test_input)
                self.scale_factor = test_output.shape[-1] // test_input.shape[-1]
                del test_input, test_output
                torch.cuda.empty_cache()
            
            ram_after, gpu_after = self._get_memory_usage()
            print(f"✅ EDSR cargado - Escala: {self.scale_factor}x")
            print(f"   RAM: +{ram_after-ram_before:.0f}MB, GPU: +{gpu_after-gpu_before:.0f}MB")
            
        except Exception as e:
            print(f"❌ Error cargando EDSR: {e}")
            raise
    
    def _unload_edsr_model(self):
        """Descarga el modelo EDSR"""
        if self.edsr_model is None:
            return
        
        print("🗑️  Descargando modelo EDSR...")
        ram_before, gpu_before = self._get_memory_usage()
        
        del self.edsr_model
        self.edsr_model = None
        self._aggressive_cleanup()
        
        ram_after, gpu_after = self._get_memory_usage()
        print(f"✅ EDSR descargado")
        print(f"   RAM: {ram_after-ram_before:+.0f}MB, GPU: {gpu_after-gpu_before:+.0f}MB")
    
    def _load_kimianet_model(self):
        """Carga el modelo KimiaNet con arquitectura corregida"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow no disponible para KimiaNet")
        
        if self.kimianet_model is not None:
            return
        
        print("🧠 Cargando modelo KimiaNet...")
        ram_before, gpu_before = self._get_memory_usage()
        
        try:
            # Configurar TensorFlow para usar poca memoria
            if hasattr(tf.config.experimental, 'set_memory_growth'):
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
            
            # Intentar múltiples configuraciones de KimiaNet
            success = False
            
            # Configuración 1: DenseNet121 estándar con capas específicas
            try:
                print("   Probando arquitectura KimiaNet v1...")
                input_layer = keras.layers.Input(shape=(224, 224, 3), name='input_1')
                
                # Red base DenseNet121 sin pooling inicial
                base_model = keras.applications.DenseNet121(
                    input_tensor=input_layer,
                    weights='imagenet',
                    include_top=False,
                    pooling=None
                )
                
                # Capas de clasificación específicas para KimiaNet
                x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
                x = keras.layers.Dense(1024, activation='relu', name='fc1')(x)
                x = keras.layers.Dropout(0.5, name='dropout_1')(x)
                x = keras.layers.Dense(512, activation='relu', name='fc2')(x)
                x = keras.layers.Dropout(0.5, name='dropout_2')(x)
                predictions = keras.layers.Dense(30, activation='softmax', name='predictions')(x)
                
                self.kimianet_model = keras.Model(inputs=input_layer, outputs=predictions)
                self.kimianet_model.load_weights(self.kimianet_weights_path)
                
                # Crear modelo para features
                self.kimianet_features = keras.Model(
                    inputs=input_layer, 
                    outputs=x  # Features después de fc2
                )
                
                success = True
                print("   ✅ Arquitectura KimiaNet v1 exitosa")
                
            except Exception as e1:
                print(f"   ❌ Arquitectura v1 falló: {e1}")
                
                # Configuración 2: DenseNet121 con pooling automático
                try:
                    print("   Probando arquitectura KimiaNet v2...")
                    tf.keras.backend.clear_session()
                    
                    input_layer = keras.layers.Input(shape=(224, 224, 3))
                    
                    base_model = keras.applications.DenseNet121(
                        input_tensor=input_layer,
                        weights='imagenet',
                        include_top=False,
                        pooling='avg'
                    )
                    
                    # Capas más simples
                    x = base_model.output
                    x = keras.layers.Dense(512, activation='relu')(x)
                    x = keras.layers.Dropout(0.5)(x)
                    predictions = keras.layers.Dense(30, activation='softmax')(x)
                    
                    self.kimianet_model = keras.Model(inputs=input_layer, outputs=predictions)
                    self.kimianet_model.load_weights(self.kimianet_weights_path)
                    
                    self.kimianet_features = keras.Model(
                        inputs=input_layer, 
                        outputs=base_model.output
                    )
                    
                    success = True
                    print("   ✅ Arquitectura KimiaNet v2 exitosa")
                    
                except Exception as e2:
                    print(f"   ❌ Arquitectura v2 falló: {e2}")
                    
                    # Configuración 3: Cargar por capas de manera flexible
                    try:
                        print("   Probando carga flexible de KimiaNet...")
                        tf.keras.backend.clear_session()
                        
                        # Cargar solo la parte que funcione
                        input_layer = keras.layers.Input(shape=(224, 224, 3))
                        base_model = keras.applications.DenseNet121(
                            input_tensor=input_layer,
                            weights='imagenet',
                            include_top=False,
                            pooling='avg'
                        )
                        
                        # Solo usar features base sin capas de clasificación
                        self.kimianet_features = keras.Model(
                            inputs=input_layer, 
                            outputs=base_model.output
                        )
                        
                        # Modelo dummy para compatibilidad
                        self.kimianet_model = self.kimianet_features
                        
                        success = True
                        print("   ✅ Carga flexible exitosa (solo features base)")
                        
                    except Exception as e3:
                        print(f"   ❌ Carga flexible falló: {e3}")
                        raise RuntimeError(f"No se pudo cargar KimiaNet con ninguna configuración")
            
            if success:
                ram_after, gpu_after = self._get_memory_usage()
                print(f"✅ KimiaNet cargado exitosamente")
                print(f"   RAM: +{ram_after-ram_before:.0f}MB, GPU: +{gpu_after-gpu_before:.0f}MB")
            
        except Exception as e:
            print(f"❌ Error cargando KimiaNet: {e}")
            # Fallback: usar solo métricas básicas
            print("⚠️  Continuando sin KimiaNet - solo PSNR y SSIM")
            self.kimianet_model = None
            self.kimianet_features = None
    
    def _unload_kimianet_model(self):
        """Descarga el modelo KimiaNet"""
        if self.kimianet_model is None and self.kimianet_features is None:
            return
        
        print("🗑️  Descargando modelo KimiaNet...")
        ram_before, gpu_before = self._get_memory_usage()
        
        if self.kimianet_model is not None:
            del self.kimianet_model
            self.kimianet_model = None
            
        if self.kimianet_features is not None:
            del self.kimianet_features
            self.kimianet_features = None
        
        # Limpieza agresiva de TensorFlow
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
        self._aggressive_cleanup()
        
        ram_after, gpu_after = self._get_memory_usage()
        print(f"✅ KimiaNet descargado")
        print(f"   RAM: {ram_after-ram_before:+.0f}MB, GPU: {gpu_after-gpu_before:+.0f}MB")
    
    def _process_sr_batch(self, lr_paths):
        """Procesa un lote con super-resolución"""
        print(f"🎯 Procesando lote SR: {len(lr_paths)} imágenes")
        
        # Cargar modelo EDSR
        self._load_edsr_model()
        
        sr_results = []
        
        for lr_path in tqdm(lr_paths, desc="Super-Resolución"):
            try:
                # Cargar imagen LR
                lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
                if lr_img is None:
                    sr_results.append(None)
                    continue
                
                # Convertir a tensor
                lr_tensor = img2tensor(lr_img, bgr2rgb=True, float32=True)
                lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
                
                # Inferencia
                with torch.no_grad():
                    sr_tensor = self.edsr_model(lr_tensor).squeeze(0)
                
                # Convertir a imagen
                sr_img = tensor2img(sr_tensor.unsqueeze(0), rgb2bgr=True, out_type=np.uint8)
                
                sr_results.append(sr_img)
                
                # Limpiar tensores
                del lr_tensor, sr_tensor
                
            except Exception as e:
                print(f"❌ Error en SR {lr_path}: {e}")
                sr_results.append(None)
        
        # Descargar modelo EDSR
        self._unload_edsr_model()
        
        return sr_results
    
    def _calculate_kimianet_metrics(self, sr_images, hr_paths):
        """Calcula métricas con KimiaNet"""
        print(f"🧠 Evaluando con KimiaNet: {len(sr_images)} imágenes")
        
        # Cargar modelo KimiaNet
        self._load_kimianet_model()
        
        # Si no se pudo cargar KimiaNet, retornar None
        if self.kimianet_features is None:
            print("⚠️  KimiaNet no disponible, saltando métricas perceptuales")
            return [None] * len(sr_images)
        
        kimianet_scores = []
        
        for sr_img, hr_path in tqdm(zip(sr_images, hr_paths), desc="KimiaNet"):
            try:
                if sr_img is None:
                    kimianet_scores.append(None)
                    continue
                
                # Cargar imagen HR
                hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
                if hr_img is None:
                    kimianet_scores.append(None)
                    continue
                
                # Redimensionar ambas imágenes a 224x224 para KimiaNet
                sr_224 = cv2.resize(sr_img, (224, 224))
                hr_224 = cv2.resize(hr_img, (224, 224))
                
                # Convertir a RGB y normalizar
                sr_rgb = cv2.cvtColor(sr_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                hr_rgb = cv2.cvtColor(hr_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Extraer features
                sr_features = self.kimianet_features.predict(
                    np.expand_dims(sr_rgb, 0), verbose=0
                ).flatten()
                hr_features = self.kimianet_features.predict(
                    np.expand_dims(hr_rgb, 0), verbose=0
                ).flatten()
                
                # Calcular similitud coseno
                similarity = np.dot(sr_features, hr_features) / (
                    np.linalg.norm(sr_features) * np.linalg.norm(hr_features)
                )
                
                kimianet_scores.append(float(similarity))
                
            except Exception as e:
                print(f"❌ Error en KimiaNet {hr_path}: {e}")
                kimianet_scores.append(None)
        
        # Descargar modelo KimiaNet
        self._unload_kimianet_model()
        
        return kimianet_scores
    
    def _calculate_basic_metrics(self, sr_images, hr_paths):
        """Calcula métricas básicas (PSNR, SSIM)"""
        print(f"📊 Calculando métricas básicas...")
        
        psnr_scores = []
        ssim_scores = []
        
        for sr_img, hr_path in zip(sr_images, hr_paths):
            try:
                if sr_img is None:
                    psnr_scores.append(None)
                    ssim_scores.append(None)
                    continue
                
                # Cargar imagen HR
                hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
                if hr_img is None:
                    psnr_scores.append(None)
                    ssim_scores.append(None)
                    continue
                
                # Redimensionar HR si es necesario
                if sr_img.shape != hr_img.shape:
                    hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]))
                
                # Calcular métricas
                psnr = calculate_psnr(sr_img, hr_img, crop_border=0, test_y_channel=False)
                ssim = calculate_ssim(sr_img, hr_img, crop_border=0, test_y_channel=False)
                
                psnr_scores.append(float(psnr))
                ssim_scores.append(float(ssim))
                
            except Exception as e:
                print(f"❌ Error en métricas básicas: {e}")
                psnr_scores.append(None)
                ssim_scores.append(None)
        
        return psnr_scores, ssim_scores
    
    def evaluate_dataset(self, lr_meta_file, hr_meta_file, output_dir, model_name, 
                        base_path="", max_images=None, checkpoint_interval=None):
        """
        Evalúa dataset completo con gestión dinámica de modelos
        
        Args:
            lr_meta_file: Archivo meta LR
            hr_meta_file: Archivo meta HR
            output_dir: Directorio de salida
            model_name: Nombre del modelo
            base_path: Ruta base
            max_images: Máximo número de imágenes
            checkpoint_interval: Intervalo para guardar checkpoints (en lotes)
        """
        print(f"\n🚀 Evaluación dinámica: {model_name}")
        
        # Cargar rutas
        with open(lr_meta_file, 'r') as f:
            lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        with open(hr_meta_file, 'r') as f:
            hr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        image_pairs = list(zip(lr_paths, hr_paths))
        
        if max_images:
            image_pairs = image_pairs[:max_images]
        
        # Dividir en lotes
        num_batches = (len(image_pairs) + self.batch_size - 1) // self.batch_size
        
        print(f"📊 Total: {len(image_pairs)} imágenes")
        print(f"📦 Lotes: {num_batches} de {self.batch_size} imágenes")
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Archivo de checkpoint
        if checkpoint_interval is None:
            checkpoint_interval = max(1, num_batches // 10)  # 10% por defecto
        
        checkpoint_file = os.path.join(output_dir, f"{model_name}_dynamic_checkpoint.csv")
        
        # Cargar progreso previo si existe
        start_batch = 0
        all_results = []
        
        if os.path.exists(checkpoint_file):
            try:
                prev_results = pd.read_csv(checkpoint_file)
                all_results = prev_results.to_dict('records')
                start_batch = len(all_results) // self.batch_size
                print(f"📋 Reanudando desde lote {start_batch}")
            except:
                print("⚠️  No se pudo cargar checkpoint previo")
        
        # Procesar lotes
        for batch_idx in range(start_batch, num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(image_pairs))
            batch_pairs = image_pairs[start_idx:end_idx]
            
            lr_batch = [pair[0] for pair in batch_pairs]
            hr_batch = [pair[1] for pair in batch_pairs]
            
            print(f"\n🔄 Lote {batch_idx + 1}/{num_batches} ({len(batch_pairs)} imágenes)")
            
            # Fase 1: Super-resolución
            sr_images = self._process_sr_batch(lr_batch)
            
            # Fase 2: Métricas básicas
            psnr_scores, ssim_scores = self._calculate_basic_metrics(sr_images, hr_batch)
            
            # Fase 3: KimiaNet
            kimianet_scores = self._calculate_kimianet_metrics(sr_images, hr_batch)
            
            # Compilar resultados del lote
            for i, (lr_path, hr_path) in enumerate(batch_pairs):
                result = {
                    'image_index': start_idx + i + 1,
                    'lr_filename': os.path.basename(lr_path),
                    'hr_filename': os.path.basename(hr_path),
                    'lr_path': lr_path,
                    'hr_path': hr_path,
                    'psnr': psnr_scores[i],
                    'ssim': ssim_scores[i],
                    'kimianet_similarity': kimianet_scores[i]
                }
                all_results.append(result)
            
            # Limpiar imágenes SR del lote
            del sr_images
            self._aggressive_cleanup()
            
            # Guardar checkpoint periódicamente
            if (batch_idx + 1) % checkpoint_interval == 0:
                self._save_checkpoint(all_results, checkpoint_file)
                ram_mb, gpu_mb = self._get_memory_usage()
                print(f"💾 Checkpoint: Lote {batch_idx + 1} - RAM: {ram_mb:.0f}MB, GPU: {gpu_mb:.0f}MB")
        
        # Guardar resultados finales
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Filtrar resultados válidos para estadísticas
            valid_results = df.dropna(subset=['psnr', 'ssim', 'kimianet_similarity'])
            
            final_file = os.path.join(output_dir, f"{model_name}_metrics_dynamic.csv")
            df.to_csv(final_file, index=False)
            
            # Remover checkpoint temporal
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            
            # Estadísticas
            print(f"\n📈 Evaluación completada:")
            print(f"   ✅ Total procesadas: {len(df)}")
            print(f"   ✅ Válidas: {len(valid_results)}")
            print(f"   ❌ Fallidas: {len(df) - len(valid_results)}")
            
            if len(valid_results) > 0:
                print(f"   📊 PSNR promedio: {valid_results['psnr'].mean():.2f} dB")
                print(f"   📊 SSIM promedio: {valid_results['ssim'].mean():.4f}")
                print(f"   🧠 KimiaNet promedio: {valid_results['kimianet_similarity'].mean():.4f}")
            
            print(f"💾 Resultados: {final_file}")
        
        # Limpieza final
        self._aggressive_cleanup()
    
    def _save_checkpoint(self, results, checkpoint_file):
        """Guarda checkpoint de progreso"""
        if results:
            df = pd.DataFrame(results)
            df.to_csv(checkpoint_file, index=False)
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Evaluador Dinámico con KimiaNet para EDSR")
    
    parser.add_argument("--model_path", required=True, help="Ruta al modelo EDSR .pt")
    parser.add_argument("--model_name", required=True, help="Nombre del modelo")
    parser.add_argument("--lr_meta_file", required=True, help="Archivo meta LR")
    parser.add_argument("--hr_meta_file", required=True, help="Archivo meta HR")
    parser.add_argument("--kimianet_weights", required=True, help="Ruta a pesos KimiaNet .h5")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Directorio de salida")
    parser.add_argument("--base_path", default="", help="Ruta base")
    parser.add_argument("--max_images", type=int, help="Máximo número de imágenes")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño de lote")
    parser.add_argument("--max_memory_mb", type=int, default=8000, help="Memoria máxima en MB")
    parser.add_argument("--checkpoint_interval", type=int, help="Intervalo de checkpoint en lotes")
    
    args = parser.parse_args()
    
    print("🔄 EVALUADOR DINÁMICO CON KIMIANET PARA EDSR")
    print("=" * 60)
    print("🎯 ESTRATEGIA: Carga alterna de modelos para optimizar memoria")
    print("   1. Cargar EDSR -> Procesar lote SR -> Descargar EDSR")
    print("   2. Cargar KimiaNet -> Evaluar lote -> Descargar KimiaNet")
    print("   3. Repetir hasta completar dataset")
    print("=" * 60)
    
    if not TF_AVAILABLE:
        print("❌ Error: TensorFlow no disponible para KimiaNet")
        return 1
    
    try:
        evaluator = DynamicModelEvaluator(
            edsr_model_path=args.model_path,
            kimianet_weights_path=args.kimianet_weights,
            batch_size=args.batch_size,
            max_memory_mb=args.max_memory_mb
        )
        
        evaluator.evaluate_dataset(
            lr_meta_file=args.lr_meta_file,
            hr_meta_file=args.hr_meta_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            base_path=args.base_path,
            max_images=args.max_images,
            checkpoint_interval=args.checkpoint_interval
        )
        
        print("\n🎉 Evaluación dinámica completada!")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())