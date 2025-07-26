#!/usr/bin/env python3
"""
Script Automatizado para Evaluación Completa de Modelos EDSR Optimizados
Ejecuta evaluación de métricas + timing GPU + timing CPU para todos los modelos .pt
"""

import os
import subprocess
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import argparse

class EDSRAutoEvaluator:
    """Evaluador automático para múltiples modelos EDSR optimizados"""
    
    def __init__(self, base_output_dir="evaluation_results", max_images=1000, num_runs=50):
        """
        Inicializa el evaluador automático
        
        Args:
            base_output_dir: Directorio base para resultados
            max_images: Máximo número de imágenes para evaluación de métricas
            num_runs: Número de runs para benchmark de timing
        """
        self.base_output_dir = base_output_dir
        self.max_images = max_images
        self.num_runs = num_runs
        self.kimianet_weights = "model-kimianet/KimiaNetKerasWeights.h5"
        
        # Configurar logging
        self.setup_logging()
        
        # Configuraciones de modelos EDSR (usando modelos .pt optimizados)
        self.model_configs = [
            {
                "name": "128to256",
                "model_path": "optimized_models/edsr_128to256_optimized.pt",
                "lr_meta": "dataset/128to256/val_lr_meta.txt",
                "hr_meta": "dataset/128to256/val_hr_meta.txt"
            },
            {
                "name": "256to512", 
                "model_path": "optimized_models/edsr_256to512_optimized.pt",
                "lr_meta": "dataset/256to512/val_lr_meta.txt",
                "hr_meta": "dataset/256to512/val_hr_meta.txt"
            },
            {
                "name": "512to1024",
                "model_path": "optimized_models/edsr_512to1024_optimized.pt", 
                "lr_meta": "dataset/512to1024/val_lr_meta.txt",
                "hr_meta": "dataset/512to1024/val_hr_meta.txt"
            },
            # Agregar más configuraciones según tus modelos disponibles
        ]
        
        # Resultados del proceso
        self.results = {
            "start_time": datetime.now().isoformat(),
            "models_evaluated": [],
            "total_duration": 0,
            "errors": []
        }
    
    def setup_logging(self):
        """Configura el sistema de logging"""
        # Crear directorio de logs
        log_dir = os.path.join(self.base_output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar logger
        log_file = os.path.join(log_dir, f"auto_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # También mostrar en consola
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Iniciando evaluación automática de modelos EDSR optimizados")
        self.logger.info(f"Log guardado en: {log_file}")
    
    def check_prerequisites(self):
        """Verifica que existan los archivos necesarios"""
        self.logger.info("🔍 Verificando prerequisites...")
        
        missing_files = []
        
        # Verificar KimiaNet weights (opcional)
        if not os.path.exists(self.kimianet_weights):
            self.logger.warning(f"KimiaNet no encontrado: {self.kimianet_weights}")
            self.logger.warning("Evaluación continuará sin índice perceptual")
        
        # Verificar archivos de cada modelo
        for config in self.model_configs:
            files_to_check = [
                config["model_path"],
                config["lr_meta"], 
                config["hr_meta"]
            ]
            
            for file_path in files_to_check:
                if not os.path.exists(file_path):
                    missing_files.append(f"{config['name']}: {file_path}")
        
        if missing_files:
            self.logger.error("❌ Archivos faltantes:")
            for file in missing_files:
                self.logger.error(f"   - {file}")
            return False
        
        self.logger.info("✅ Todos los archivos necesarios están disponibles")
        return True
    
    def run_command(self, command, description, env_vars=None, show_progress=True):
        """
        Ejecuta un comando y maneja errores
        
        Args:
            command: Lista con el comando a ejecutar
            description: Descripción del comando para logging
            env_vars: Variables de entorno adicionales
            show_progress: Si mostrar la salida en tiempo real
            
        Returns:
            Tuple (success, duration, output)
        """
        self.logger.info(f"🚀 Ejecutando: {description}")
        self.logger.info(f"   Comando: {' '.join(command)}")
        
        # Preparar entorno
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            self.logger.info(f"   Variables de entorno: {env_vars}")
        
        start_time = time.time()
        
        try:
            if show_progress:
                # Mostrar salida en tiempo real
                self.logger.info(f"📺 Mostrando progreso en tiempo real...")
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                    universal_newlines=True
                )
                
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    # Mostrar línea en tiempo real
                    print(line.rstrip())
                    output_lines.append(line)
                
                process.wait()
                result_code = process.returncode
                output = ''.join(output_lines)
                
            else:
                # Modo silencioso (capturar todo)
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=3600
                )
                result_code = result.returncode
                output = result.stdout if result.returncode == 0 else result.stderr
            
            duration = time.time() - start_time
            
            if result_code == 0:
                self.logger.info(f"✅ {description} completado en {duration:.1f}s")
                return True, duration, output
            else:
                self.logger.error(f"❌ {description} falló:")
                self.logger.error(f"   Error código: {result_code}")
                if not show_progress:
                    self.logger.error(f"   Salida: {output}")
                return False, duration, output
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.logger.error(f"⏰ {description} excedió el timeout ({duration:.1f}s)")
            return False, duration, "Timeout"
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"💥 Error ejecutando {description}: {e}")
            return False, duration, str(e)
    
    def evaluate_metrics(self, config):
        """Ejecuta evaluación de métricas para un modelo"""
        model_name = f"EDSR_{config['name']}"
        output_dir = os.path.join(self.base_output_dir, config['name'], "Validation_Images")
        
        command = [
            "python", "scripts/evaluate_edsr_validation.py",
            "--model_path", config["model_path"],
            "--model_name", model_name,
            "--lr_meta_file", config["lr_meta"],
            "--hr_meta_file", config["hr_meta"],
            "--output_dir", output_dir,
            "--max_images", str(self.max_images)
        ]
        
        # Agregar KimiaNet si está disponible
        command.extend(["--use_kimianet", "--kimianet_weights", self.kimianet_weights])
        
        return self.run_command(
            command, 
            f"Evaluación de métricas - {config['name']}",
            show_progress=True  # Mostrar progreso en tiempo real
        )
    
    def benchmark_timing_gpu(self, config):
        """Ejecuta benchmark de timing en GPU para un modelo"""
        model_name = f"EDSR_{config['name']}_GPU"
        output_dir = os.path.join(self.base_output_dir, config['name'], "realistic_timing_results")
        
        command = [
            "python", "scripts/benchmark_edsr_timing.py",
            "--model_path", config["model_path"],
            "--model_name", model_name,
            "--lr_meta_file", config["lr_meta"],
            "--num_runs", str(self.num_runs),
            "--output_dir", output_dir,
            "--device", "gpu"
        ]
        
        return self.run_command(
            command,
            f"Benchmark timing GPU - {config['name']}",
            show_progress=True  # Mostrar progreso en tiempo real
        )
    
    def benchmark_timing_cpu(self, config):
        """Ejecuta benchmark de timing en CPU para un modelo"""
        model_name = f"EDSR_{config['name']}_CPU"
        output_dir = os.path.join(self.base_output_dir, config['name'], "realistic_timing_results")
        
        command = [
            "python", "scripts/benchmark_edsr_timing.py",
            "--model_path", config["model_path"],
            "--model_name", model_name,
            "--lr_meta_file", config["lr_meta"],
            "--num_runs", str(self.num_runs),
            "--output_dir", output_dir,
            "--device", "cpu"
        ]
        
        # Forzar CPU con variable de entorno
        env_vars = {"CUDA_VISIBLE_DEVICES": ""}
        
        return self.run_command(
            command,
            f"Benchmark timing CPU - {config['name']}",
            env_vars,
            show_progress=True  # Mostrar progreso en tiempo real
        )
    
    def evaluate_single_model(self, config):
        """Evalúa un modelo completo (métricas + timing GPU + timing CPU)"""
        model_name = config["name"]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 INICIANDO EVALUACIÓN: {model_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        model_start_time = time.time()
        model_results = {
            "model_name": model_name,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        # Crear directorio de salida
        model_output_dir = os.path.join(self.base_output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Paso 1: Evaluación de métricas
        self.logger.info(f"\n📊 PASO 1/3: Evaluación de métricas - {model_name}")
        success, duration, output = self.evaluate_metrics(config)
        
        step_result = {
            "step": "metrics_evaluation",
            "success": success,
            "duration": duration,
            "output": output[:500] if output else ""  # Limitar output para JSON
        }
        model_results["steps"].append(step_result)
        
        if not success:
            self.logger.error(f"❌ Evaluación de métricas falló para {model_name}")
            model_results["total_duration"] = time.time() - model_start_time
            return model_results
        
        # Paso 2: Benchmark timing GPU
        self.logger.info(f"\n🚀 PASO 2/3: Benchmark timing GPU - {model_name}")
        success, duration, output = self.benchmark_timing_gpu(config)
        
        step_result = {
            "step": "timing_gpu",
            "success": success,
            "duration": duration,
            "output": output[:500] if output else ""
        }
        model_results["steps"].append(step_result)
        
        if not success:
            self.logger.error(f"❌ Benchmark GPU falló para {model_name}")
        
        # Paso 3: Benchmark timing CPU (siempre ejecutar, aunque GPU falle)
        self.logger.info(f"\n🖥️  PASO 3/3: Benchmark timing CPU - {model_name}")
        success, duration, output = self.benchmark_timing_cpu(config)
        
        step_result = {
            "step": "timing_cpu",
            "success": success,
            "duration": duration,
            "output": output[:500] if output else ""
        }
        model_results["steps"].append(step_result)
        
        if not success:
            self.logger.error(f"❌ Benchmark CPU falló para {model_name}")
        
        # Finalizar evaluación del modelo
        model_results["total_duration"] = time.time() - model_start_time
        model_results["end_time"] = datetime.now().isoformat()
        
        successful_steps = sum(1 for step in model_results["steps"] if step["success"])
        total_steps = len(model_results["steps"])
        
        self.logger.info(f"\n🏁 MODELO {model_name.upper()} COMPLETADO")
        self.logger.info(f"   Pasos exitosos: {successful_steps}/{total_steps}")
        self.logger.info(f"   Duración total: {model_results['total_duration']:.1f}s")
        
        return model_results
    
    def run_full_evaluation(self, models_to_evaluate=None):
        """
        Ejecuta evaluación completa de todos los modelos
        
        Args:
            models_to_evaluate: Lista de nombres de modelos a evaluar, o None para todos
        """
        self.logger.info(f"\n🚀 INICIANDO EVALUACIÓN AUTOMÁTICA COMPLETA")
        self.logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        # Verificar prerequisites
        if not self.check_prerequisites():
            self.logger.error("❌ Prerequisites no cumplidos. Abortando evaluación.")
            return
        
        # Filtrar modelos si se especifica
        configs_to_run = self.model_configs
        if models_to_evaluate:
            configs_to_run = [c for c in self.model_configs if c["name"] in models_to_evaluate]
            self.logger.info(f"📋 Evaluando modelos específicos: {models_to_evaluate}")
        else:
            self.logger.info(f"📋 Evaluando todos los modelos: {[c['name'] for c in configs_to_run]}")
        
        # Evaluar cada modelo
        for i, config in enumerate(configs_to_run, 1):
            self.logger.info(f"\n🎯 MODELO {i}/{len(configs_to_run)}: {config['name']}")
            
            try:
                model_results = self.evaluate_single_model(config)
                self.results["models_evaluated"].append(model_results)
                
            except Exception as e:
                self.logger.error(f"💥 Error crítico evaluando modelo {config['name']}: {e}")
                error_info = {
                    "model": config["name"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.results["errors"].append(error_info)
        
        # Finalizar evaluación
        self.results["total_duration"] = time.time() - start_time
        self.results["end_time"] = datetime.now().isoformat()
        
        # Guardar resumen de resultados
        self.save_summary()
        
        # Mostrar resumen final
        self.show_final_summary()
    
    def save_summary(self):
        """Guarda un resumen de todos los resultados"""
        summary_file = os.path.join(self.base_output_dir, "evaluation_summary.json")
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            self.logger.info(f"📄 Resumen guardado en: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando resumen: {e}")
    
    def show_final_summary(self):
        """Muestra resumen final de la evaluación"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🏆 RESUMEN FINAL DE EVALUACIÓN")
        self.logger.info(f"{'='*80}")
        
        total_models = len(self.results["models_evaluated"])
        total_duration = self.results["total_duration"]
        
        self.logger.info(f"📊 Modelos evaluados: {total_models}")
        self.logger.info(f"⏱️  Duración total: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        
        # Resumen por modelo
        for model_result in self.results["models_evaluated"]:
            model_name = model_result["model_name"]
            successful_steps = sum(1 for step in model_result["steps"] if step["success"])
            total_steps = len(model_result["steps"])
            duration = model_result["total_duration"]
            
            status = "✅" if successful_steps == total_steps else "⚠️" if successful_steps > 0 else "❌"
            
            self.logger.info(f"   {status} {model_name}: {successful_steps}/{total_steps} pasos ({duration:.1f}s)")
        
        # Errores si los hay
        if self.results["errors"]:
            self.logger.info(f"\n❌ Errores encontrados: {len(self.results['errors'])}")
            for error in self.results["errors"]:
                self.logger.info(f"   - {error['model']}: {error['error']}")
        
        self.logger.info(f"\n📂 Resultados disponibles en: {self.base_output_dir}")
        self.logger.info(f"🎉 EVALUACIÓN AUTOMÁTICA COMPLETADA")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Evaluación Automática de Modelos EDSR Optimizados")
    
    parser.add_argument(
        "--models",
        nargs='+',
        choices=['128to256', '256to512', '512to1024'],
        help="Modelos específicos a evaluar (default: todos)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="Directorio base para resultados"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=1000,
        help="Máximo número de imágenes para evaluación de métricas"
    )
    
    parser.add_argument(
        "--num_runs", 
        type=int,
        default=50,
        help="Número de runs para benchmark de timing"
    )
    
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="Verificar configuración sin ejecutar comandos"
    )
    
    args = parser.parse_args()
    
    print("🤖 EVALUADOR AUTOMÁTICO DE MODELOS EDSR OPTIMIZADOS")
    print("=" * 50)
    
    if args.dry_run:
        print("🔍 MODO DRY RUN - Solo verificación")
    
    # Crear evaluador
    evaluator = EDSRAutoEvaluator(
        base_output_dir=args.output_dir,
        max_images=args.max_images,
        num_runs=args.num_runs
    )
    
    if args.dry_run:
        # Solo verificar prerequisites
        if evaluator.check_prerequisites():
            print("✅ Configuración válida. Todo listo para ejecutar.")
            print("\nComandos que se ejecutarían:")
            for config in evaluator.model_configs:
                if not args.models or config["name"] in args.models:
                    print(f"\n📦 {config['name']}:")
                    print(f"   1. Evaluación métricas")
                    print(f"   2. Timing GPU")
                    print(f"   3. Timing CPU")
        else:
            print("❌ Configuración inválida. Revisa los archivos faltantes.")
    else:
        # Ejecutar evaluación completa
        try:
            evaluator.run_full_evaluation(args.models)
        except KeyboardInterrupt:
            print("\n⏹️  Evaluación interrumpida por el usuario")
        except Exception as e:
            print(f"\n💥 Error crítico: {e}")

if __name__ == "__main__":
    main()