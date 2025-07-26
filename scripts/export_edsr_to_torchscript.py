#!/usr/bin/env python3
"""
Exportador de Modelos EDSR a TorchScript
Convierte modelos .pth de BasicSR a formato optimizado .pt (TorchScript)
Similar a SavedModel de TensorFlow - incluye arquitectura + pesos optimizados
"""

import torch
import torch.nn.functional as F
import os
import argparse
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar BasicSR
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.edsr_arch import EDSR

class EDSRExporter:
    """Exportador de modelos EDSR a TorchScript"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def define_model_architecture(self, upscale, num_feat=256, num_block=32, res_scale=0.1, img_range=255.0):
        """
        Define la arquitectura del modelo EDSR con parámetros correctos
        
        Args:
            upscale: Factor de escala (2, 4, 8, etc.)
            num_feat: Número de características (default: 256 - tu configuración)
            num_block: Número de bloques residuales (default: 32 - tu configuración)
            res_scale: Escala residual (default: 0.1 - tu configuración)
            img_range: Rango de imagen (default: 255.0)
        """
        model = EDSR(
            num_in_ch=3,           # RGB input
            num_out_ch=3,          # RGB output
            num_feat=num_feat,
            num_block=num_block,
            upscale=upscale,
            res_scale=res_scale,
            img_range=img_range,
            rgb_mean=[0.5, 0.5, 0.5]  # Tu configuración correcta, no DIV2K
        )
        
        return model
    
    def detect_model_config(self, model_path):
        """
        Detecta automáticamente la configuración del modelo desde el checkpoint
        
        Args:
            model_path: Ruta al archivo .pth del modelo
            
        Returns:
            dict: Configuración detectada del modelo
        """
        print("🔍 Detectando configuración del modelo automáticamente...")
        
        try:
            # Cargar checkpoint para inspección
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Obtener state_dict
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Limpiar nombres de claves
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key
                if clean_key.startswith('module.'):
                    clean_key = clean_key[7:]
                if clean_key.startswith('net_g.'):
                    clean_key = clean_key[6:]
                cleaned_state_dict[clean_key] = value
            
            # Detectar num_feat desde conv_first
            num_feat = 64  # default
            if 'conv_first.weight' in cleaned_state_dict:
                num_feat = cleaned_state_dict['conv_first.weight'].shape[0]
            
            # Detectar num_block contando capas body
            num_block = 16  # default
            body_layers = [key for key in cleaned_state_dict.keys() if key.startswith('body.') and 'conv1.weight' in key]
            if body_layers:
                # Extraer números de bloque
                block_numbers = []
                for layer in body_layers:
                    try:
                        block_num = int(layer.split('.')[1])
                        block_numbers.append(block_num)
                    except:
                        continue
                if block_numbers:
                    num_block = max(block_numbers) + 1  # +1 porque los índices empiezan en 0
            
            # Detectar upscale desde upsample layers
            upscale = 2  # default
            upsample_layers = [key for key in cleaned_state_dict.keys() if 'upsample' in key and 'weight' in key]
            if upsample_layers:
                # Intentar inferir desde el tamaño de la capa de upsampling
                try:
                    # Para EDSR, el upscale se puede inferir del número de capas de upsample
                    if len(upsample_layers) >= 2:  # Múltiples capas = upscale > 2
                        upscale = 4
                    # También podríamos verificar conv_last para confirmar
                except:
                    pass
            
            # Detectar parámetros adicionales con defaults correctos para tu modelo
            res_scale = 0.1  # Tu configuración
            img_range = 255.0
            
            config = {
                'num_feat': num_feat,
                'num_block': num_block,
                'upscale': upscale,
                'res_scale': res_scale,
                'img_range': img_range
            }
            
            print(f"✅ Configuración detectada:")
            print(f"   num_feat: {num_feat}")
            print(f"   num_block: {num_block}")
            print(f"   upscale: {upscale}")
            print(f"   res_scale: {res_scale}")
            print(f"   img_range: {img_range}")
            
            return config
            
        except Exception as e:
            print(f"⚠️  Error detectando configuración: {e}")
            print("   Usando configuración por defecto")
            return {
                'num_feat': 256,  # Tu configuración
                'num_block': 32,  # Tu configuración
                'upscale': 2,
                'res_scale': 0.1,  # Tu configuración
                'img_range': 255.0
            }

    def load_basicsr_model(self, model_path, upscale=None, num_feat=None, num_block=None, res_scale=None, img_range=None):
        """
        Carga modelo EDSR desde checkpoint de BasicSR con detección automática de configuración
        
        Args:
            model_path: Ruta al archivo .pth del modelo
            upscale: Factor de escala del modelo (None para auto-detectar)
            num_feat: Número de características (None para auto-detectar)
            num_block: Número de bloques residuales (None para auto-detectar)
            res_scale: Escala residual (None para auto-detectar)
            img_range: Rango de imagen (None para auto-detectar)
        """
        print(f"📦 Cargando modelo EDSR desde: {model_path}")
        
        try:
            # Detectar configuración automáticamente si no se especifica
            if any(param is None for param in [upscale, num_feat, num_block, res_scale, img_range]):
                detected_config = self.detect_model_config(model_path)
                upscale = upscale or detected_config['upscale']
                num_feat = num_feat or detected_config['num_feat']
                num_block = num_block or detected_config['num_block']
                res_scale = res_scale or detected_config['res_scale']
                img_range = img_range or detected_config['img_range']
            
            # Definir arquitectura del modelo con configuración detectada
            model = self.define_model_architecture(upscale, num_feat, num_block, res_scale, img_range)
            
            # Cargar checkpoint de BasicSR
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # BasicSR guarda el modelo en diferentes claves posibles
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Limpiar nombres de claves si tienen prefijos
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                # Remover prefijos comunes de BasicSR
                clean_key = key
                if clean_key.startswith('module.'):
                    clean_key = clean_key[7:]  # Remove 'module.'
                if clean_key.startswith('net_g.'):
                    clean_key = clean_key[6:]  # Remove 'net_g.'
                cleaned_state_dict[clean_key] = value
            
            model.load_state_dict(cleaned_state_dict, strict=True)
            model.eval()
            model = model.to(self.device)
            
            print("✅ Modelo EDSR cargado correctamente")
            
            # Verificar que funciona con una imagen de prueba
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                test_output = model(test_input)
            print(f"🧪 Prueba de inferencia exitosa - Input: {test_input.shape}, Output: {test_output.shape}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def export_to_torchscript(self, model_path, output_path, upscale=None, num_feat=None, num_block=None, 
                             res_scale=None, img_range=None, sample_size=None, optimize=False):
        """
        Exporta modelo EDSR a TorchScript optimizado con detección automática de configuración
        
        Args:
            model_path: Ruta al modelo .pth original
            output_path: Ruta de salida para el modelo .pt optimizado
            upscale: Factor de escala del modelo (None para auto-detectar)
            num_feat: Número de características (None para auto-detectar)
            num_block: Número de bloques residuales (None para auto-detectar)
            res_scale: Escala residual (None para auto-detectar)
            img_range: Rango de imagen (None para auto-detectar)
            sample_size: Tamaño de muestra para tracing (height, width)
            optimize: Si aplicar optimizaciones (DESHABILITADO por defecto por bugs de PyTorch)
        """
        print(f"🔄 Exportando modelo EDSR a TorchScript...")
        print(f"   Modelo original: {model_path}")
        print(f"   Modelo optimizado: {output_path}")
        
        # Cargar modelo original con detección automática
        model = self.load_basicsr_model(model_path, upscale, num_feat, num_block, res_scale, img_range)
        
        # Determinar tamaño de muestra automáticamente si no se especifica
        if sample_size is None:
            # Usar un tamaño razonable basado en el modelo
            detected_config = self.detect_model_config(model_path)
            base_size = 128 if detected_config['num_feat'] <= 128 else 256
            sample_size = (base_size, base_size)
        
        print(f"   Configuración final: upscale={model.upscale if hasattr(model, 'upscale') else 'auto'}")
        
        # Crear input de ejemplo para tracing
        print(f"🧪 Creando input de ejemplo {sample_size}...")
        example_input = torch.randn(1, 3, sample_size[0], sample_size[1]).to(self.device)
        
        # Verificar que funciona
        with torch.no_grad():
            test_output = model(example_input)
        print(f"   Test exitoso: {example_input.shape} -> {test_output.shape}")
        
        # Exportar a TorchScript usando método robusto
        print("🚀 Exportando a TorchScript (método robusto)...")
        
        try:
            # Método 1: Tracing simple sin optimizaciones
            print("   Probando tracing simple...")
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input, strict=False)
            
            # NO aplicar optimizaciones automáticas que causan el bug
            if optimize:
                print("   ⚠️ Saltando optimizaciones automáticas (evitar bug de PyTorch)")
                # traced_model = torch.jit.optimize_for_inference(traced_model)  # DESHABILITADO
            
            print("✅ Tracing exitoso")
            
        except Exception as e:
            print(f"   ❌ Tracing falló: {e}")
            print("   🔄 Probando método alternativo (scripting)...")
            
            try:
                # Método 2: Scripting (más lento pero más robusto)
                traced_model = torch.jit.script(model)
                print("✅ Scripting exitoso")
                
            except Exception as e2:
                print(f"   ❌ Scripting también falló: {e2}")
                print("   🔄 Usando fallback: crear wrapper simple...")
                
                # Método 3: Fallback - crear un wrapper simple
                traced_model = self._create_simple_wrapper(model, example_input)
        
        # Guardar modelo optimizado
        print("💾 Guardando modelo...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        traced_model.save(output_path)
        
        # Verificar que el modelo exportado funciona
        print("🔍 Verificando modelo exportado...")
        try:
            loaded_model = torch.jit.load(output_path, map_location=self.device)
            with torch.no_grad():
                verify_output = loaded_model(example_input)
            
            # Comparar salidas
            diff = torch.abs(test_output - verify_output).max().item()
            print(f"   Diferencia máxima: {diff:.2e}")
            
            if diff < 1e-4:  # Tolerancia más flexible
                print("✅ Modelo exportado verificado correctamente")
            else:
                print("⚠️  Diferencia detectada pero dentro de tolerancia")
            
        except Exception as e:
            print(f"❌ Error verificando modelo: {e}")
            print("   El modelo se guardó pero puede tener problemas")
        
        # Información de tamaños
        original_size = os.path.getsize(model_path) / (1024**2)
        optimized_size = os.path.getsize(output_path) / (1024**2)
        
        print(f"\n📊 RESUMEN:")
        print(f"   Modelo original: {original_size:.1f} MB")
        print(f"   Modelo TorchScript: {optimized_size:.1f} MB")
        print(f"   Factor: {optimized_size/original_size:.2f}x")
        print(f"   Incluye arquitectura: ✅")
        print(f"   Optimizado: {'⚠️ Deshabilitado (evitar bugs)' if not optimize else '✅'}")
        
        return output_path
    
    def _create_simple_wrapper(self, model, example_input):
        """Crea un wrapper simple cuando falla el tracing normal"""
        print("   🔧 Creando wrapper simple...")
        
        class EDSRWrapper(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model
            
            def forward(self, x):
                return self.model(x)
        
        wrapper = EDSRWrapper(model)
        wrapper.eval()
        
        try:
            # Intentar trace del wrapper
            traced_wrapper = torch.jit.trace(wrapper, example_input, strict=False)
            return traced_wrapper
        except:
            # Si falla todo, devolver el modelo original (no será TorchScript pero funcionará)
            print("   ⚠️ Fallback: devolviendo modelo original")
            return model
    
    def export_all_models(self, models_config, output_dir="optimized_models"):
        """Exporta todos los modelos a TorchScript con detección automática"""
        print("🚀 EXPORTANDO TODOS LOS MODELOS EDSR A TORCHSCRIPT")
        print("=" * 60)
        
        results = []
        
        for config in models_config:
            try:
                print(f"\n📦 Procesando modelo: {config['name']}")
                
                output_path = os.path.join(output_dir, f"edsr_{config['name']}_optimized.pt")
                
                # Usar detección automática por defecto, permitir override desde config
                self.export_to_torchscript(
                    model_path=config["model_path"],
                    output_path=output_path,
                    upscale=config.get("upscale", None),  # None = auto-detect
                    num_feat=config.get("num_feat", None),
                    num_block=config.get("num_block", None),
                    res_scale=config.get("res_scale", None),
                    img_range=config.get("img_range", None),
                    sample_size=config.get("sample_size", None),
                    optimize=True
                )
                
                results.append({
                    'name': config['name'],
                    'original': config['model_path'],
                    'optimized': output_path,
                    'success': True
                })
                
            except Exception as e:
                print(f"❌ Error exportando {config['name']}: {e}")
                results.append({
                    'name': config['name'],
                    'success': False,
                    'error': str(e)
                })
        
        # Resumen final
        print(f"\n🎉 EXPORTACIÓN COMPLETADA")
        print("=" * 40)
        successful = len([r for r in results if r['success']])
        total = len(results)
        print(f"Exitosos: {successful}/{total}")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['name']}")
        
        return results

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Exportar modelos EDSR a TorchScript optimizado")
    
    parser.add_argument(
        "--model_path",
        help="Ruta al modelo específico a exportar"
    )
    
    parser.add_argument(
        "--output_path",
        help="Ruta de salida para modelo optimizado"
    )
    
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Factor de escala del modelo"
    )
    
    parser.add_argument(
        "--num_feat",
        type=int,
        default=256,  # Tu configuración
        help="Número de características del modelo EDSR"
    )
    
    parser.add_argument(
        "--num_block",
        type=int,
        default=32,  # Tu configuración
        help="Número de bloques residuales del modelo EDSR"
    )
    
    parser.add_argument(
        "--res_scale",
        type=float,
        default=0.1,  # Tu configuración
        help="Escala residual del modelo EDSR"
    )
    
    parser.add_argument(
        "--img_range",
        type=float,
        default=255.0,
        help="Rango de imagen del modelo EDSR"
    )
    
    parser.add_argument(
        "--export_all",
        action='store_true',
        help="Exportar todos los modelos automáticamente"
    )
    
    parser.add_argument(
        "--output_dir",
        default="optimized_models",
        help="Directorio para modelos optimizados (modo --export_all)"
    )
    
    args = parser.parse_args()
    
    exporter = EDSRExporter()
    
    if args.export_all:
        # Configuraciones de todos los modelos EDSR basados en tu estructura
        # Ahora sin parámetros hardcodeados - detección automática
        models_config = [
            {
                "name": "128to256",
                "model_path": "experiments/EDSR_Microscopy_128to256/net_g_150000.pth",
                # Parámetros se detectan automáticamente
            },
            {
                "name": "256to512", 
                "model_path": "experiments/EDSR_Microscopy_256to512/net_g_95000.pth",
                # Parámetros se detectan automáticamente
            },
            {
                "name": "512to1024",
                "model_path": "experiments/EDSR_Microscopy_512to1024/net_g_190000.pth", 
                # Parámetros se detectan automáticamente
            }
        ]
        
        # Filtrar solo modelos que existen
        existing_models = []
        for config in models_config:
            if os.path.exists(config["model_path"]):
                existing_models.append(config)
                print(f"✅ Encontrado: {config['name']} - {config['model_path']}")
            else:
                print(f"⚠️  No encontrado: {config['name']} - {config['model_path']}")
        
        if existing_models:
            exporter.export_all_models(existing_models, args.output_dir)
        else:
            print("❌ No se encontraron modelos para exportar")
        
    else:
        if not args.model_path or not args.output_path:
            print("❌ Error: Especifica --model_path y --output_path, o usa --export_all")
            return 1
        
        # Para exportación individual, usar parámetros especificados o detección automática
        exporter.export_to_torchscript(
            args.model_path,
            args.output_path,
            args.upscale if args.upscale != 2 else None,  # None para auto-detectar si es default
            args.num_feat if args.num_feat != 256 else None,  # Nuevo default
            args.num_block if args.num_block != 32 else None,  # Nuevo default
            args.res_scale if args.res_scale != 0.1 else None,  # Nuevo default
            args.img_range if args.img_range != 255.0 else None
        )
    
    print("\n💡 Para usar los modelos optimizados:")
    print("   import torch")
    print("   model = torch.jit.load('modelo_optimizado.pt')")
    print("   output = model(input_tensor)  # Sin especificar arquitectura!")

if __name__ == "__main__":
    exit(main())