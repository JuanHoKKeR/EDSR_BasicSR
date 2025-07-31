# EDSR for Histopathology Super-Resolution

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)
[![BasicSR](https://img.shields.io/badge/BasicSR-framework-green.svg)](https://github.com/XPixelGroup/BasicSR)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Una implementación especializada de **Enhanced Deep Super-Resolution Network (EDSR)** optimizada para imágenes de histopatología de cáncer de mama. Este proyecto forma parte de un **Trabajo de Grado** enfocado en super-resolución para aplicaciones médicas mediante arquitecturas CNN residuales.

## 🎯 **Objetivo del Proyecto**

Adaptar y evaluar EDSR, una arquitectura CNN residual consolidada y estable, para super-resolución de imágenes de microscopia histopatológica, aprovechando su simplicidad arquitectural y robustez en el entrenamiento.

## ✨ **Características Principales**

- **🔬 Especializado en Histopatología**: Optimizado para imágenes de cáncer de mama
- **🏗️ Arquitectura CNN Residual**: Utiliza bloques residuales profundos para mejor reconstrucción
- **⚡ Estabilidad Excepcional**: Entrenamiento robusto sin mode collapse o inestabilidades
- **🎯 Simplicidad Efectiva**: Arquitectura limpia y bien documentada
- **📊 Referencia Confiable**: Excelente baseline para comparaciones experimentales
- **📈 Sistema de Evaluación Comprehensive**: Métricas especializadas para imágenes médicas

## 🔄 **Diferencias con el Proyecto Original**

Este repositorio está basado en [BasicSR](https://github.com/XPixelGroup/BasicSR) pero incluye adaptaciones específicas:

| Aspecto | BasicSR Original | Esta Implementación |
|---------|------------------|-------------------|
| **Dominio** | Imágenes naturales (DIV2K) | Histopatología específica |
| **Dataset** | Datasets estándar | Dataset histopatológico especializado |
| **Configuración** | Configuraciones generales | Optimizadas para imágenes médicas |
| **Evaluación** | Métricas básicas | Sistema comprehensive médico |
| **Entrenamiento** | Multi-escala estándar | Enfoque en factores específicos |
| **Aplicación** | Uso general | Diagnóstico médico asistido |

## 🚀 **Inicio Rápido**

### Prerequisitos
- Python 3.9+
- PyTorch 1.13+
- CUDA 11.2+ (recomendado)
- GPU con 8GB+ VRAM para entrenamiento

### 1. Clonar el Repositorio
```bash
git clone https://github.com/JuanHoKKeR/EDSR_BasicSR.git
cd EDSR_BasicSR
```

### 2. Instalación de Dependencias
```bash
# Instalar BasicSR y dependencias
pip install basicsr
pip install -r requirements.txt

# Instalar en modo desarrollo (recomendado)
python setup.py develop
```

### 3. Preparar el Dataset
Organiza tu dataset con la siguiente estructura:
```
datasets/
├── histopatologia/
│   ├── train/
│   │   ├── hr/              # Imágenes de alta resolución
│   │   └── lr/              # Imágenes de baja resolución
│   ├── val/
│   │   ├── hr/
│   │   └── lr/
│   └── test/
│       ├── hr/
│       └── lr/
```

### 4. Configurar el Entrenamiento
Edita el archivo de configuración YAML en `options/train/EDSR/`:

```yaml
# Configuración general
name: EDSR_histopatologia_x2_f256_b32
model_type: SRModel
scale: 2
num_gpu: 1
manual_seed: 0

# Dataset configuration
datasets:
  train:
    name: histopatologia_train
    type: PairedImageDataset
    dataroot_gt: datasets/histopatologia/train/hr
    dataroot_lq: datasets/histopatologia/train/lr
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    gt_size: 256
    use_hflip: true
    use_rot: true
    
    # Data loader
    use_shuffle: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 4
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: histopatologia_val
    type: PairedImageDataset
    dataroot_gt: datasets/histopatologia/val/hr
    dataroot_lq: datasets/histopatologia/val/lr
    io_backend:
      type: disk

# Network structures
network_g:
  type: EDSR
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 256           # Número de características
  num_block: 32           # Número de bloques residuales
  upscale: 2
  res_scale: 0.1
  img_range: 255.
  rgb_mean: [0.4488, 0.4371, 0.4040]

# Training settings
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [200000, 400000, 600000, 800000]
    gamma: 0.5

  total_iter: 1000000
  warmup_iter: -1  # Sin warmup

  # Losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean
```

### 5. Ejecutar Entrenamiento
```bash
# Entrenamiento básico
python basicsr/train.py -opt options/train/EDSR/train_EDSR_histopatologia_x2.yml

# Con logging detallado
python basicsr/train.py -opt options/train/EDSR/train_EDSR_histopatologia_x2.yml --debug
```

## 📁 **Estructura del Proyecto**

```
EDSR_BasicSR/
├── basicsr/                        # Framework BasicSR
│   ├── archs/                      # Arquitecturas de redes
│   │   ├── edsr_arch.py           # Implementación EDSR
│   │   └── arch_util.py           # Utilidades de arquitectura
│   ├── data/                       # Manejo de datasets
│   │   ├── paired_image_dataset.py # Dataset pareado LR-HR
│   │   └── transforms.py          # Transformaciones de datos
│   ├── models/                     # Modelos de entrenamiento
│   │   ├── sr_model.py            # Modelo base de super-resolución
│   │   └── base_model.py          # Modelo base
│   ├── losses/                     # Funciones de pérdida
│   │   ├── basic_loss.py          # L1, L2, etc.
│   │   └── perceptual_loss.py     # Pérdida perceptual
│   └── train.py                   # Script principal de entrenamiento
├── options/                        # Archivos de configuración
│   ├── train/EDSR/                # Configuraciones de entrenamiento
│   │   ├── train_EDSR_histopatologia_x2.yml
│   │   ├── train_EDSR_histopatologia_x4.yml
│   │   └── train_EDSR_custom.yml
│   └── test/EDSR/                 # Configuraciones de testing
├── experiments/                    # Resultados experimentales
│   └── [experiment_name]/
│       ├── models/                # Checkpoints del modelo
│       ├── training_states/       # Estados del optimizador
│       ├── visualization/         # Imágenes de validación
│       └── train_[timestamp].log  # Logs de entrenamiento
├── datasets/                       # Datasets organizados
├── results/                        # Resultados de testing
└── requirements.txt               # Dependencias
```

## 🧠 **Arquitectura EDSR**

### **Enhanced Deep Super-Resolution Network**

EDSR elimina las capas de normalización batch de ResNet original y utiliza:

```python
# Arquitectura base en basicsr/archs/edsr_arch.py
class EDSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=256, 
                 num_block=32, upscale=2, res_scale=0.1, img_range=255.):
        
        # Extracción de características inicial
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Bloques residuales principales
        self.body = make_layer(ResidualBlockNoBN, num_block, 
                              num_feat=num_feat, res_scale=res_scale)
        
        # Convolución antes del upsampling
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.upsample = Upsample(upscale, num_feat)
        
        # Reconstrucción final
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
```

### **Configuraciones por Resolución**

#### **128→256 (×2) - Configuración Estándar**
```yaml
network_g:
  type: EDSR
  num_feat: 256
  num_block: 32
  upscale: 2
  res_scale: 0.1

train:
  total_iter: 300000
  batch_size_per_gpu: 16
```

#### **256→512 (×2) - Configuración Intermedia**
```yaml
network_g:
  type: EDSR
  num_feat: 256
  num_block: 32
  upscale: 2
  res_scale: 0.1

train:
  total_iter: 500000
  batch_size_per_gpu: 8
```

#### **512→1024 (×2) - Alta Resolución**
```yaml
network_g:
  type: EDSR
  num_feat: 256
  num_block: 32
  upscale: 2
  res_scale: 0.1

train:
  total_iter: 800000
  batch_size_per_gpu: 4
```

## 🚀 **Scripts Principales**

### **1. Entrenamiento**

#### Entrenamiento Básico
```bash
python basicsr/train.py -opt options/train/EDSR/train_EDSR_histopatologia_x2.yml
```

#### Entrenamiento con Validación Automática
```bash
python basicsr/train.py \
    -opt options/train/EDSR/train_EDSR_histopatologia_x2.yml \
    --auto_resume \
    --debug
```

#### Reanudar Entrenamiento
```bash
python basicsr/train.py \
    -opt options/train/EDSR/train_EDSR_histopatologia_x2.yml \
    --auto_resume
```

### **2. Testing y Evaluación**

#### Testing Básico
```bash
python basicsr/test.py -opt options/test/EDSR/test_EDSR_histopatologia_x2.yml
```

#### Configuración de Testing
```yaml
# test_EDSR_histopatologia_x2.yml
name: EDSR_histopatologia_x2_test
model_type: SRModel
scale: 2
num_gpu: 1

datasets:
  test_1:
    name: histopatologia_test
    type: PairedImageDataset
    dataroot_gt: datasets/histopatologia/test/hr
    dataroot_lq: datasets/histopatologia/test/lr
    io_backend:
      type: disk

network_g:
  type: EDSR
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 256
  num_block: 32
  upscale: 2
  res_scale: 0.1
  img_range: 255.

path:
  pretrain_network_g: experiments/EDSR_histopatologia_x2/models/net_g_latest.pth
  strict_load_g: true

val:
  save_img: true
  suffix: ~  # Sin sufijo en nombre de archivo
  
  metrics:
    psnr: # Peak signal-to-noise ratio
      type: calculate_psnr
      crop_border: 2
      test_y_channel: false
      
    ssim: # Structural similarity
      type: calculate_ssim
      crop_border: 2
      test_y_channel: false
```

## ⚙️ **Configuración Avanzada**

### **Optimización de Memoria**

Para GPUs con memoria limitada:

```yaml
# Reducir batch size
train:
  batch_size_per_gpu: 2  # En lugar de 4 o más

# Usar gradient checkpointing (si está disponible)
network_g:
  type: EDSR
  use_checkpoint: true  # Para ahorrar memoria

# Ajustar workers
datasets:
  train:
    num_worker_per_gpu: 4  # Reducir si hay problemas de memoria
```

### **Aceleración de Entrenamiento**

```yaml
# Usar precisión mixta (AMP)
train:
  use_amp: true
  
# Optimizar DataLoader
datasets:
  train:
    prefetch_mode: cuda  # Precarga en GPU
    pin_memory: true
    
# Scheduler optimizado
train:
  scheduler:
    type: CosineAnnealingRestartLR
    periods: [250000, 250000, 250000, 250000]
    restart_weights: [1, 0.5, 0.5, 0.5]
    eta_min: !!float 1e-7
```

### **Data Augmentation para Histopatología**

```yaml
datasets:
  train:
    # Augmentaciones básicas
    use_hflip: true      # Flip horizontal
    use_rot: true        # Rotaciones 90°
    
    # Augmentaciones de color (cuidadoso en histopatología)
    color_jitter_prob: 0.1
    color_jitter_shift: 20
    
    # Cropping inteligente
    gt_size: 256
    crop_type: center    # center, random
    
    # Normalización específica para histopatología
    mean: [0.485, 0.456, 0.406]  # Valores típicos
    std: [0.229, 0.224, 0.225]
```

## 📊 **Resultados y Rendimiento**

### **Modelos Implementados**

| Modelo | Resolución | Parámetros | Tiempo Entrenamiento* | Estabilidad |
|--------|------------|------------|----------------------|-------------|
| 128→256 | 128×128 → 256×256 | ~43M | ~2 días | ✅ Excelente |
| 256→512 | 256×256 → 512×512 | ~43M | ~3 días | ✅ Excelente |
| 512→1024 | 512×512 → 1024×1024 | ~43M | ~4 días | ✅ Excelente |

*Tiempo estimado en RTX 4090

### **Características Técnicas**

#### **Ventajas de EDSR:**
✅ **Arquitectura probada**: CNN residual consolidada y estable  
✅ **Entrenamiento robusto**: Sin problemas de convergencia  
✅ **Implementación limpia**: Código bien estructurado y mantenible  
✅ **Reproducibilidad**: Resultados consistentes entre ejecuciones  
✅ **Escalabilidad**: Mantiene arquitectura constante entre resoluciones  
✅ **Simplicidad**: Sin componentes adversariales complejos  

#### **Características Técnicas:**
🔧 **Función de pérdida**: L1 Loss únicamente  
🔧 **Optimizador**: Adam con learning rate scheduling  
🔧 **Normalización**: Sin Batch Normalization (clave del diseño)  
🔧 **Activación**: ReLU en bloques residuales  
🔧 **Upsampling**: Sub-pixel convolution (PixelShuffle)  

### **Comparación de Estabilidad**

| Aspecto | EDSR | ESRGAN | SwinIR |
|---------|------|--------|--------|
| **Convergencia** | ✅ Monótona | ⚠️ Oscilante | ✅ Estable |
| **Tiempo entrenamiento** | ⚠️ Largo | ✅ Moderado | ⚠️ Largo |
| **Reproducibilidad** | ✅ Excelente | ⚠️ Variable | ✅ Buena |
| **Simplicidad** | ✅ Máxima | ⚠️ Compleja | ⚠️ Compleja |

## 🔧 **Herramientas y Utilidades**

### **Monitoreo de Entrenamiento**

```bash
# Visualizar logs
tensorboard --logdir experiments/EDSR_histopatologia_x2/tb_logger

# Métricas automáticas disponibles:
# - l_pix: Pérdida L1
# - psnr: PSNR en validación
# - ssim: SSIM en validación
# - lr: Learning rate actual
```

### **Gestión de Experimentos**

```yaml
# Configuración de logging
logger:
  print_freq: 100           # Frecuencia de logs en consola
  save_checkpoint_freq: 5000 # Guardar modelo cada 5k iter
  use_tb_logger: true       # TensorBoard logging
  wandb:                    # Weights & Biases (opcional)
    project: EDSR_histopatologia
    resume_id: ~
```

### **Validación Automática**

```yaml
# Validación durante entrenamiento
val:
  val_freq: !!float 5e3    # Validar cada 5k iteraciones
  save_img: false          # No guardar imágenes (ahorra espacio)
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 2
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 2
      test_y_channel: false
```

## 🏗️ **Personalización y Extensiones**

### **Modificar Arquitectura**

```python
# En basicsr/archs/edsr_arch.py
class CustomEDSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=256, 
                 num_block=32, upscale=2, res_scale=0.1):
        
        # Modificar número de características
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Ajustar número de bloques residuales
        self.body = make_layer(ResidualBlockNoBN, num_block, 
                              num_feat=num_feat, res_scale=res_scale)
```

### **Funciones de Pérdida Personalizadas**

```python
# En basicsr/losses/custom_loss.py
class HistopathologyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        # Pérdida L1 base
        l1_loss = self.l1_loss(pred, target)
        
        # Agregar pérdidas específicas para histopatología
        # (preservación de bordes, texturas, etc.)
        
        return l1_loss * self.loss_weight
```

## 🎯 **Casos de Uso y Aplicaciones**

### **Diagnóstico Médico Asistido**
- Mejora de imágenes histopatológicas de baja calidad
- Preparación de imágenes para análisis automatizado
- Standardización de calidad en diferentes equipos de microscopía

### **Investigación Científica**
- Baseline confiable para comparaciones experimentales
- Arquitectura de referencia para nuevos métodos
- Validación de técnicas de evaluación

### **Análisis de Estructuras Celulares**
- Preservación fiel de morfología tisular
- Reconstrucción estable de patrones histológicos
- Análisis detallado de arquitectura glandular

## 🤝 **Contribución**

Este proyecto es parte de un Trabajo de Grado enfocado en super-resolución médica. Las contribuciones son bienvenidas siguiendo las mejores prácticas de BasicSR.

## 📄 **Licencia**

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 **Reconocimientos**

- **Framework Original**: [BasicSR](https://github.com/XPixelGroup/BasicSR) por XPixel Group
- **EDSR Paper**: Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution." CVPRW 2017.
- **ResNet**: He, Kaiming, et al. "Deep residual learning for image recognition." CVPR 2016.
- **BasicSR Team**: Por el excelente framework de super-resolución

## 📞 **Contacto**

- **Autor**: Juan David Cruz Useche
- **Proyecto**: Trabajo de Grado - Super-Resolución para Histopatología
- **GitHub**: [@JuanHoKKeR](https://github.com/JuanHoKKeR)
- **Repositorio**: [EDSR_BasicSR](https://github.com/JuanHoKKeR/EDSR_BasicSR)

## 📚 **Referencias**

```bibtex
@inproceedings{lim2017enhanced,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}

@misc{wang2020basicsr,
  title={BasicSR: Open Source Image and Video Restoration Toolbox},
  author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
  howpublished={\url{https://github.com/XPixelGroup/BasicSR}},
  year={2020}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

---

**⭐ Si este proyecto te resulta útil para tu investigación en super-resolución médica, considera darle una estrella!**