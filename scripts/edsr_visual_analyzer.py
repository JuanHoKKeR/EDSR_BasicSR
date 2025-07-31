#!/usr/bin/env python3
"""
Analizador Visual de Resultados EDSR
Compara imagen LR, predicción EDSR, HR original y genera análisis visual completo
Adaptado del script original de ESRGAN para usar PyTorch y EDSR con el mismo diseño de gráficas
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar las librerías necesarias para KimiaNet (si están disponibles)
try:
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow no disponible. Índice perceptual no estará disponible.")
    TF_AVAILABLE = False

# Importar pytorch-msssim si está disponible
try:
    from pytorch_msssim import ssim, ms_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    print("⚠️  pytorch-msssim no disponible. MS-SSIM no estará disponible.")
    MSSSIM_AVAILABLE = False

# Configurar fuente Computer Modern Roman para matplotlib
plt.rcParams.update({
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'font.size': 11
})


class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet+KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no está disponible")
            
        print("🧠 Cargando DenseNet121 con pesos KimiaNet...")
        
        # Cargar DenseNet121 sin la capa final
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,
            input_shape=(None, None, 3)
        )
        
        # Cargar pesos KimiaNet si existe el archivo
        if kimianet_weights_path and os.path.exists(kimianet_weights_path):
            try:
                self.densenet.load_weights(kimianet_weights_path)
                print(f"✅ Pesos KimiaNet cargados desde: {kimianet_weights_path}")
            except Exception as e:
                print(f"⚠️  Error cargando pesos KimiaNet: {e}")
                print("    Usando DenseNet121 sin preentrenar")
        else:
            print("⚠️  No se encontraron pesos KimiaNet, usando DenseNet121 sin preentrenar")
        
        # Usar una capa intermedia para extraer características
        try:
            feature_layer = self.densenet.get_layer('conv4_block6_concat')
        except:
            try:
                feature_layer = self.densenet.get_layer('conv4_block24_concat')
            except:
                feature_layer = self.densenet.layers[-2]
        
        self.feature_extractor = tf.keras.Model(
            inputs=self.densenet.input,
            outputs=feature_layer.output
        )
        
        # Congelar el modelo  
        for layer in self.feature_extractor.layers:
            layer.trainable = False
            
        print(f"✅ Extractor de características listo: {feature_layer.name}")
    
    def calculate_perceptual_distance(self, img1, img2):
        """Calcula distancia perceptual entre dos imágenes usando KimiaNet"""
        # Asegurar que sean tensores float32
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)
        
        # Agregar dimensión de batch si no existe
        if len(img1.shape) == 3:
            img1 = tf.expand_dims(img1, 0)
        if len(img2.shape) == 3:
            img2 = tf.expand_dims(img2, 0)
        
        # Normalizar para DenseNet
        img1_norm = (img1 - 127.5) / 127.5  # [-1, 1]
        img2_norm = (img2 - 127.5) / 127.5  # [-1, 1]
        
        # Extraer características
        features1 = self.feature_extractor(img1_norm)
        features2 = self.feature_extractor(img2_norm)
        
        # Calcular distancia L2 entre características
        perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
        
        return perceptual_distance


class EDSRVisualAnalyzer:
    """Analizador visual completo de resultados de superresolución para EDSR"""
    
    def __init__(self, kimianet_weights_path=None):
        """Inicializa el analizador"""
        self.perceptual_evaluator = None
        if kimianet_weights_path and TF_AVAILABLE:
            try:
                self.perceptual_evaluator = KimiaNetPerceptualLoss(kimianet_weights_path)
            except Exception as e:
                print(f"⚠️  Error inicializando evaluador perceptual: {e}")
    
    def load_image(self, image_path, target_size=None):
        """Carga una imagen usando OpenCV y la convierte a tensor PyTorch"""
        try:
            # Leer archivo con OpenCV (BGR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a float32 y mantener rango [0, 255] como el original
            image = image.astype(np.float32)
            
            # Convertir a tensor PyTorch
            image_tensor = torch.from_numpy(image)
            
            if target_size is not None:
                # Cambiar formato para interpolación: HWC -> CHW -> NCHW
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                image_tensor = F.interpolate(image_tensor, size=target_size, mode='bicubic', align_corners=False)
                # Volver a HWC
                image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
            
            return image_tensor
        except Exception as e:
            print(f"❌ Error cargando {image_path}: {e}")
            return None
    
    def resize_to_match(self, img1, img2):
        """Redimensiona img2 para que coincida con img1 si es necesario"""
        if img1.shape != img2.shape:
            print(f"⚠️  Redimensionando: {img2.shape} → {img1.shape}")
            # HWC -> CHW -> NCHW para interpolación
            img2 = img2.permute(2, 0, 1).unsqueeze(0)
            img2 = F.interpolate(img2, size=img1.shape[:2], mode='bicubic', align_corners=False)
            # Volver a HWC
            img2 = img2.squeeze(0).permute(1, 2, 0)
        return img2
    
    def calculate_psnr(self, img1, img2):
        """Calcula PSNR entre dos imágenes usando PyTorch"""
        # Asegurar que las imágenes tengan el mismo tamaño
        if img1.shape != img2.shape:
            img2 = self.resize_to_match(img1, img2)
        
        # Calcular MSE
        mse = torch.mean((img1 - img2) ** 2)
        
        # Calcular PSNR (usando rango 255 para imágenes uint8)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse))
        return float(psnr)
    
    def calculate_ssim_metrics(self, img1, img2):
        """Calcula SSIM y MS-SSIM usando pytorch-msssim"""
        if not MSSSIM_AVAILABLE:
            return {'ssim': None, 'ms_ssim': None}
        
        # Convertir a formato [1, 3, H, W] para pytorch-msssim
        def to_tensor_format(img):
            if len(img.shape) == 3:
                # Normalizar a [0, 1] para SSIM
                img_norm = img / 255.0
                return img_norm.permute(2, 0, 1).unsqueeze(0)
            return img
        
        img1_tensor = to_tensor_format(img1)
        img2_tensor = to_tensor_format(img2)
        
        # Redimensionar si es necesario
        if img1_tensor.shape != img2_tensor.shape:
            img2_tensor = F.interpolate(img2_tensor, size=img1_tensor.shape[2:], mode='bicubic', align_corners=False)
        
        try:
            # Calcular SSIM
            ssim_val = ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
            
            # Calcular MS-SSIM
            ms_ssim_val = ms_ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
            
            return {
                'ssim': float(ssim_val),
                'ms_ssim': float(ms_ssim_val)
            }
        except Exception as e:
            print(f"⚠️  Error calculando SSIM: {e}")
            return {'ssim': None, 'ms_ssim': None}
    
    def calculate_metrics(self, predicted_img, ground_truth_img):
        """Calcula todas las métricas de evaluación"""
        
        # Asegurar mismo tamaño
        if predicted_img.shape != ground_truth_img.shape:
            print(f"⚠️  Redimensionando ground truth: {ground_truth_img.shape} → {predicted_img.shape}")
            ground_truth_img = self.resize_to_match(predicted_img, ground_truth_img)
        
        # PSNR
        psnr = self.calculate_psnr(predicted_img, ground_truth_img)
        
        # SSIM y MS-SSIM
        ssim_metrics = self.calculate_ssim_metrics(predicted_img, ground_truth_img)
        
        # MSE y MAE
        mse = torch.mean((predicted_img - ground_truth_img) ** 2)
        mae = torch.mean(torch.abs(predicted_img - ground_truth_img))
        
        metrics = {
            'psnr': float(psnr),
            'ssim': ssim_metrics['ssim'],
            'ms_ssim': ssim_metrics['ms_ssim'],
            'mse': float(mse),
            'mae': float(mae)
        }
        
        # Índice perceptual si está disponible
        if self.perceptual_evaluator:
            try:
                # Convertir tensores PyTorch a numpy para TensorFlow
                pred_np = predicted_img.numpy().astype(np.uint8)
                gt_np = ground_truth_img.numpy().astype(np.uint8)
                
                perceptual_dist = self.perceptual_evaluator.calculate_perceptual_distance(
                    pred_np, gt_np
                )
                metrics['perceptual_index'] = float(perceptual_dist.numpy())
            except Exception as e:
                print(f"⚠️  Error calculando índice perceptual: {e}")
                metrics['perceptual_index'] = None
        else:
            metrics['perceptual_index'] = None
        
        return metrics
    
    def calculate_absolute_difference(self, predicted_img, ground_truth_img):
        """Calcula diferencia absoluta normalizada"""
        # Normalizar a [0, 1]
        pred_norm = predicted_img / 255.0
        gt_norm = ground_truth_img / 255.0
        
        # Diferencia absoluta
        abs_diff = torch.abs(pred_norm - gt_norm)
        
        return abs_diff
    
    def generate_latex_table(self, metrics, output_path, image_name=""):
        """Genera tabla LaTeX con las métricas"""
        
        scale_factor = "Determinado por tamaños de imagen"
        
        latex_table = f"""\\begin{{table}}[!htb]
\\centering
\\caption{{Métricas de evaluación para superresolución EDSR: {image_name}}}
\\label{{tab:edsr_visual_metrics_{image_name.lower().replace(' ', '_')}}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Valor}} \\\\
\\hline
PSNR (dB) & {metrics['psnr']:.4f} \\\\
"""
        
        if metrics['ssim'] is not None:
            latex_table += f"SSIM & {metrics['ssim']:.4f} \\\\\n"
        
        if metrics['ms_ssim'] is not None:
            latex_table += f"MS-SSIM & {metrics['ms_ssim']:.4f} \\\\\n"
        
        latex_table += f"""MSE & {metrics['mse']:.2f} \\\\
MAE & {metrics['mae']:.2f} \\\\
"""
        
        if metrics['perceptual_index'] is not None:
            latex_table += f"Índice Perceptual & {metrics['perceptual_index']:.6f} \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\textbf{Notas:} PSNR más alto indica mejor fidelidad. SSIM evalúa similitud estructural (0-1, más alto mejor). MS-SSIM es extensión multi-escala de SSIM. MSE y MAE miden error medio cuadrático y absoluto respectivamente. Índice Perceptual basado en KimiaNet (más bajo mejor).
\\end{table}
"""
        
        # Guardar archivo LaTeX
        latex_file = output_path.replace('.png', '_metrics.tex').replace('.jpg', '_metrics.tex')
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"📄 Tabla LaTeX guardada en: {latex_file}")
        return latex_file
    
    def create_comprehensive_analysis(self, lr_path, predicted_path, hr_path, output_path, image_name="Análisis EDSR"):
        """
        Crea análisis visual completo de resultados de superresolución EDSR
        Mantiene el mismo diseño que el script original de ESRGAN
        """
        
        print("🎨 CREANDO ANÁLISIS VISUAL COMPLETO EDSR")
        print("=" * 60)
        
        # Cargar imágenes
        print("📁 Cargando imágenes...")
        lr_img = self.load_image(lr_path)
        predicted_img = self.load_image(predicted_path)
        hr_img = self.load_image(hr_path)
        
        if any(img is None for img in [lr_img, predicted_img, hr_img]):
            print("❌ Error cargando alguna de las imágenes")
            return None
        
        print(f"   LR: {lr_img.shape}")
        print(f"   Predicha: {predicted_img.shape}")
        print(f"   HR: {hr_img.shape}")
        
        # Asegurar que HR y predicha tengan el mismo tamaño
        if predicted_img.shape != hr_img.shape:
            print(f"⚠️  Redimensionando HR: {hr_img.shape} → {predicted_img.shape}")
            hr_img = self.resize_to_match(predicted_img, hr_img)
        
        # Calcular factor de escala
        scale_factor = predicted_img.shape[0] // lr_img.shape[0]
        
        # Calcular métricas
        print("📊 Calculando métricas...")
        metrics = self.calculate_metrics(predicted_img, hr_img)
        
        # Mostrar métricas
        print(f"\n📈 MÉTRICAS CALCULADAS:")
        print(f"   PSNR: {metrics['psnr']:.4f} dB")
        if metrics['ssim'] is not None:
            print(f"   SSIM: {metrics['ssim']:.4f}")
        if metrics['ms_ssim'] is not None:
            print(f"   MS-SSIM: {metrics['ms_ssim']:.4f}")
        print(f"   MSE: {metrics['mse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        if metrics['perceptual_index'] is not None:
            print(f"   Índice Perceptual: {metrics['perceptual_index']:.6f}")
        
        # Calcular diferencia absoluta
        abs_diff = self.calculate_absolute_difference(predicted_img, hr_img)
        
        # Generar tabla LaTeX
        self.generate_latex_table(metrics, output_path, image_name)
        
        # Crear visualización usando el MISMO diseño que el original de ESRGAN
        print("🎨 Generando visualización...")
        
        fig = plt.figure(figsize=(18, 5))  # Misma configuración que ESRGAN
        
        # Convertir tensores a numpy para matplotlib
        def tensor_to_numpy(tensor):
            return np.clip(tensor.numpy().astype(np.uint8), 0, 255)
        
        # Título principal idéntico al original pero cambiando "Superresolución" por "Superresolución EDSR"
        fig.suptitle(f'Análisis Completo de Superresolución EDSR: {image_name}', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Configurar subplot EXACTAMENTE igual que el original
        gs = GridSpec(1, 5, figure=fig, width_ratios=[0.5, 0.3, 2, 2, 2], 
                     wspace=0.2, left=0.05, right=0.95, top=0.85, bottom=0.05)
        
        # 1. Imagen LR (MUCHO más pequeña visualmente) - IDÉNTICO al original
        ax1 = fig.add_subplot(gs[0, 0])
        lr_display = tensor_to_numpy(lr_img)
        ax1.imshow(lr_display)
        ax1.set_title(f'LR Input\n{lr_img.shape[0]}×{lr_img.shape[1]}', 
                     fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Agregar borde rojo MÁS GRUESO para destacar - IDÉNTICO
        rect = patches.Rectangle((0, 0), lr_img.shape[1]-1, lr_img.shape[0]-1, 
                               linewidth=4, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # Agregar texto "PEQUEÑA" para enfatizar - IDÉNTICO
        ax1.text(0.5, -0.15, '¡Pequeña!', ha='center', va='top', 
                fontsize=9, fontweight='bold', color='red',
                transform=ax1.transAxes)
        
        # 2. Flecha y texto de modelo - CAMBIO MÍNIMO: "EDSR" en lugar de "MODELO SR"
        ax_arrow = fig.add_subplot(gs[0, 1])
        ax_arrow.axis('off')
        
        # Flecha más dramática - IDÉNTICO
        arrow = patches.FancyArrowPatch((0.1, 0.5), (0.9, 0.5),
                                       connectionstyle="arc3", 
                                       arrowstyle='->', mutation_scale=30,
                                       transform=ax_arrow.transAxes, 
                                       color='blue', linewidth=4)
        ax_arrow.add_patch(arrow)
        
        # Texto del modelo - ÚNICO CAMBIO: EDSR en lugar de MODELO SR
        ax_arrow.text(0.5, 0.7, f'EDSR\n×{scale_factor}', ha='center', va='center', 
                     fontsize=9, fontweight='bold', color='blue',
                     transform=ax_arrow.transAxes)
        
        # 3. Imagen predicha (GRANDE) - CAMBIO MÍNIMO: "Predicción EDSR"
        ax2 = fig.add_subplot(gs[0, 2])
        predicted_display = tensor_to_numpy(predicted_img)
        ax2.imshow(predicted_display)
        ax2.set_title(f'Predicción EDSR\n{predicted_img.shape[0]}×{predicted_img.shape[1]}', 
                     fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Agregar texto "GRANDE" para contrastar - IDÉNTICO
        ax2.text(0.5, -0.08, '¡Grande!', ha='center', va='top', 
                fontsize=10, fontweight='bold', color='green',
                transform=ax2.transAxes)
        
        # 4. Imagen HR original (GRANDE) - IDÉNTICO
        ax3 = fig.add_subplot(gs[0, 3])
        hr_display = tensor_to_numpy(hr_img)
        ax3.imshow(hr_display)
        ax3.set_title(f'HR Original\n{hr_img.shape[0]}×{hr_img.shape[1]}', 
                     fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # 5. Diferencia absoluta con mapa de calor - IDÉNTICO
        ax4 = fig.add_subplot(gs[0, 4])
        diff_display = torch.mean(abs_diff, dim=-1).numpy()  # Promedio de canales RGB
        
        im = ax4.imshow(diff_display, cmap='hot', vmin=0, vmax=0.3)
        ax4.set_title(f'Diferencia Absoluta\nError Promedio: {np.mean(diff_display):.4f}', 
                     fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # Colorbar para diferencia absoluta - IDÉNTICO
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Error Absoluto', rotation=270, labelpad=12, fontsize=9)
        
        # Ajustar layout compacto - IDÉNTICO
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Guardar imagen - IDÉNTICO
        print(f"💾 Guardando visualización en: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()
        
        # Resumen en consola - CAMBIO MÍNIMO: "EDSR" en el título
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL ANÁLISIS EDSR")
        print("=" * 60)
        
        # Determinar factor de escala - IDÉNTICO
        scale_factor = predicted_img.shape[0] // lr_img.shape[0]
        print(f"🔍 Factor de escala detectado: ×{scale_factor}")
        print(f"📏 Resolución: {lr_img.shape[0]}×{lr_img.shape[1]} → {predicted_img.shape[0]}×{predicted_img.shape[1]}")
        
        # Evaluación de calidad - IDÉNTICO
        if metrics['psnr'] > 25:
            quality = "Excelente"
        elif metrics['psnr'] > 22:
            quality = "Buena"
        elif metrics['psnr'] > 18:
            quality = "Moderada"
        else:
            quality = "Baja"
        
        print(f"⭐ Calidad estimada: {quality}")
        print(f"💡 Error promedio: {np.mean(diff_display):.4f}")
        
        if metrics['perceptual_index'] is not None:
            if metrics['perceptual_index'] < 0.01:
                perceptual_quality = "Excelente fidelidad biológica"
            elif metrics['perceptual_index'] < 0.05:
                perceptual_quality = "Buena fidelidad biológica"
            else:
                perceptual_quality = "Fidelidad biológica moderada"
            print(f"🧠 Evaluación perceptual: {perceptual_quality}")
        
        print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Analizador visual completo de resultados EDSR (diseño idéntico al original de ESRGAN)")
    
    parser.add_argument("--lr", required=True, help="Imagen LR de entrada")
    parser.add_argument("--predicted", required=True, help="Imagen predicha por EDSR")
    parser.add_argument("--hr", required=True, help="Imagen HR original (ground truth)")
    parser.add_argument("--output", required=True, help="Ruta de salida para el análisis")
    parser.add_argument("--kimianet_weights", 
                       default=None,
                       help="Pesos KimiaNet para índice perceptual")
    parser.add_argument("--name", default="Análisis EDSR", help="Nombre para el análisis")
    
    args = parser.parse_args()
    
    # Verificar que todos los archivos existen
    required_files = [args.lr, args.predicted, args.hr]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ No existe el archivo: {file_path}")
            return 1
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Crear analizador
        analyzer = EDSRVisualAnalyzer(args.kimianet_weights)
        
        # Realizar análisis
        metrics = analyzer.create_comprehensive_analysis(
            lr_path=args.lr,
            predicted_path=args.predicted,
            hr_path=args.hr,
            output_path=args.output,
            image_name=args.name
        )
        
        if metrics:
            print(f"✅ Análisis visual EDSR completado: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())