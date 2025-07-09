import os
import argparse
import re
import random
from collections import defaultdict

def extract_info(filepath):
    """Extrae información del nombre del archivo para emparejar imágenes usando el número de parche"""
    # Extraer el nombre del archivo de la ruta completa
    filename = os.path.basename(filepath)
    
    # Buscar el patrón que incluye las coordenadas x, y y el número de parche
    # Formato esperado: *_x####_y####_Patch###.jpg
    match = re.search(r'_x(\d+)_y(\d+)_Patch(\d+)\.jpg$', filename)
    if match:
        x_coord = match.group(1)
        y_coord = match.group(2)
        patch_num = match.group(3)
        
        # Crear una clave única basada en coordenadas y número de parche
        unique_key = f"x{x_coord}_y{y_coord}_Patch{patch_num}"
        return unique_key
    return None

def verify_pair_alignment(lr_file, hr_file):
    """Verifica que los archivos LR y HR mantengan el emparejamiento correcto"""
    with open(lr_file, 'r') as f:
        lr_lines = [line.strip() for line in f if line.strip()]
    
    with open(hr_file, 'r') as f:
        hr_lines = [line.strip() for line in f if line.strip()]
    
    if len(lr_lines) != len(hr_lines):
        print(f"❌ ERROR: Diferentes números de líneas: LR={len(lr_lines)}, HR={len(hr_lines)}")
        return False
    
    mismatches = 0
    for i, (lr_path, hr_path) in enumerate(zip(lr_lines, hr_lines)):
        lr_key = extract_info(lr_path)
        hr_key = extract_info(hr_path)
        
        if lr_key != hr_key:
            mismatches += 1
            if mismatches <= 3:  # Mostrar solo los primeros 3 errores
                print(f"❌ Mismatch en línea {i+1}:")
                print(f"   LR: {os.path.basename(lr_path)} -> {lr_key}")
                print(f"   HR: {os.path.basename(hr_path)} -> {hr_key}")
    
    if mismatches == 0:
        print(f"✅ VERIFICACIÓN EXITOSA: {len(lr_lines)} pares perfectamente alineados")
        return True
    else:
        print(f"❌ VERIFICACIÓN FALLIDA: {mismatches} pares mal alineados de {len(lr_lines)}")
        return False

def create_meta_pairs(hr_meta_file, lr_meta_file, output_dir, sample_size=None, 
                     shuffle_variants=1, verify_alignment=True, seed=42):
    """Crea archivos meta_info con pares LR-HR consistentes basados en el número de parche"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Leer archivos meta_info
    hr_paths = []
    with open(hr_meta_file, 'r') as f:
        hr_paths = [line.strip() for line in f if line.strip()]
    
    lr_paths = []
    with open(lr_meta_file, 'r') as f:
        lr_paths = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Dataset inicial:")
    print(f"   HR images: {len(hr_paths)}")
    print(f"   LR images: {len(lr_paths)}")
    
    # Organizar imágenes por su identificador único de parche
    hr_dict = {}
    hr_failed = 0
    for path in hr_paths:
        unique_key = extract_info(path)
        if unique_key:
            hr_dict[unique_key] = path
        else:
            hr_failed += 1
    
    lr_dict = {}
    lr_failed = 0
    for path in lr_paths:
        unique_key = extract_info(path)
        if unique_key:
            lr_dict[unique_key] = path
        else:
            lr_failed += 1
    
    if hr_failed > 0 or lr_failed > 0:
        print(f"⚠️  Archivos con nombres no reconocidos: HR={hr_failed}, LR={lr_failed}")
    
    # Encontrar identificadores comunes (parches que existen en ambas resoluciones)
    common_keys = set(hr_dict.keys()).intersection(set(lr_dict.keys()))
    print(f"✅ Found {len(common_keys)} common patches")
    
    # Crear pares de imágenes
    pairs = []
    for key in common_keys:
        lr_path = lr_dict[key]
        hr_path = hr_dict[key]
        pairs.append((lr_path, hr_path))
    
    print(f"✅ Created {len(pairs)} exact LR-HR pairs based on patch number")
    
    # Análisis de distribución
    if pairs:
        print(f"\n📊 Análisis de distribución:")
        
        # Analizar distribución por coordenadas para detectar posibles sesgos
        coord_distribution = defaultdict(int)
        for lr_path, hr_path in pairs:
            key = extract_info(lr_path)
            if key:
                # Extraer solo las coordenadas para análisis
                coord_part = re.search(r'x(\d+)_y(\d+)', key)
                if coord_part:
                    coord_key = f"x{coord_part.group(1)}_y{coord_part.group(2)}"
                    coord_distribution[coord_key] += 1
        
        unique_coords = len(coord_distribution)
        print(f"   Coordenadas únicas: {unique_coords}")
        print(f"   Promedio parches por coordenada: {len(pairs)/unique_coords:.1f}")
    
    # Muestra aleatoria si se solicita
    if sample_size and sample_size < len(pairs):
        print(f"\n🎯 Sampling {sample_size} pairs randomly")
        # Usar seed fijo para reproducibilidad
        random.seed(seed)
        random.shuffle(pairs)
        pairs = pairs[:sample_size]
    
    # Generar múltiples variantes con diferentes shuffles
    all_outputs = []
    
    for variant in range(shuffle_variants):
        print(f"\n🔀 Generando variante {variant + 1}/{shuffle_variants}")
        
        # Shuffle con seed diferente para cada variante
        variant_pairs = pairs.copy()
        random.seed(seed + variant)
        random.shuffle(variant_pairs)
        
        # Nombres de archivos para esta variante
        if shuffle_variants == 1:
            suffix = ""
        else:
            suffix = f"_shuffle{variant + 1}"
        
        paired_hr_meta = os.path.join(output_dir, f"paired_hr_meta{suffix}.txt")
        paired_lr_meta = os.path.join(output_dir, f"paired_lr_meta{suffix}.txt")
        
        # Escribir archivos meta_info de pares
        with open(paired_hr_meta, 'w') as f:
            for pair in variant_pairs:
                f.write(f"{pair[1]}\n")
        
        with open(paired_lr_meta, 'w') as f:
            for pair in variant_pairs:
                f.write(f"{pair[0]}\n")
        
        print(f"💾 Saved paired meta_info files:")
        print(f"   HR: {paired_hr_meta}")
        print(f"   LR: {paired_lr_meta}")
        
        # Verificar alineamiento si se solicita
        if verify_alignment:
            print(f"🔍 Verificando alineamiento...")
            verify_pair_alignment(paired_lr_meta, paired_hr_meta)
        
        all_outputs.append((paired_lr_meta, paired_hr_meta))
        
        # Mostrar algunos ejemplos de pares para verificación (solo primera variante)
        if variant == 0 and variant_pairs:
            print(f"\n📋 Ejemplos de pares emparejados:")
            for i in range(min(3, len(variant_pairs))):
                lr_key = extract_info(variant_pairs[i][0])
                hr_key = extract_info(variant_pairs[i][1])
                print(f"   Par {i+1}:")
                print(f"     LR: {os.path.basename(variant_pairs[i][0])} -> {lr_key}")
                print(f"     HR: {os.path.basename(variant_pairs[i][1])} -> {hr_key}")
                print(f"     ✅ Match: {lr_key == hr_key}")
    
    # Generar archivo de configuración sugerida
    config_suggestion = os.path.join(output_dir, "suggested_config.yaml")
    with open(config_suggestion, 'w') as f:
        f.write("# Configuración sugerida para dataset pre-shuffleado\n")
        f.write("# Con dataset ya mezclado, usar buffer MÍNIMO\n\n")
        f.write(f"dataset_info:\n")
        f.write(f"  total_pairs: {len(pairs)}\n")
        f.write(f"  shuffle_variants: {shuffle_variants}\n")
        f.write(f"  pre_shuffled: true\n\n")
        f.write("# Configuración TensorFlow optimizada\n")
        f.write("tensorflow_config:\n")
        f.write("  shuffle_buffer: 1  # MÍNIMO porque dataset ya está mezclado\n")
        f.write("  prefetch_buffer: 1  # MÍNIMO para ahorrar memoria\n")
        f.write("  num_parallel_calls: 2  # Reducido para 256→512\n\n")
        f.write("# Para implementar en load_dataset_from_meta_info:\n")
        f.write("# dataset = dataset.shuffle(1, reshuffle_each_iteration=False)\n")
    
    print(f"\n💡 Configuración sugerida guardada en: {config_suggestion}")
    
    return all_outputs

def create_dataset_splits(paired_lr_file, paired_hr_file, output_dir, 
                         splits={'train': 0.8, 'val': 0.15, 'test': 0.05}, seed=42):
    """Crear splits de entrenamiento/validación/test manteniendo el emparejamiento"""
    
    print(f"\n📂 Creando splits del dataset...")
    
    # Leer pares
    with open(paired_lr_file, 'r') as f:
        lr_paths = [line.strip() for line in f if line.strip()]
    
    with open(paired_hr_file, 'r') as f:
        hr_paths = [line.strip() for line in f if line.strip()]
    
    if len(lr_paths) != len(hr_paths):
        print(f"❌ ERROR: Archivos desalineados")
        return
    
    # Crear índices shuffleados
    indices = list(range(len(lr_paths)))
    random.seed(seed)
    random.shuffle(indices)
    
    # Calcular tamaños de splits
    total = len(indices)
    train_size = int(total * splits['train'])
    val_size = int(total * splits['val'])
    test_size = total - train_size - val_size
    
    print(f"   Total: {total}")
    print(f"   Train: {train_size} ({splits['train']*100:.1f}%)")
    print(f"   Val: {val_size} ({splits['val']*100:.1f}%)")
    print(f"   Test: {test_size} ({(test_size/total)*100:.1f}%)")
    
    # Crear splits
    splits_indices = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:]
    }
    
    # Guardar splits
    for split_name, split_indices in splits_indices.items():
        lr_split_file = os.path.join(output_dir, f"{split_name}_lr_meta.txt")
        hr_split_file = os.path.join(output_dir, f"{split_name}_hr_meta.txt")
        
        with open(lr_split_file, 'w') as f:
            for idx in split_indices:
                f.write(f"{lr_paths[idx]}\n")
        
        with open(hr_split_file, 'w') as f:
            for idx in split_indices:
                f.write(f"{hr_paths[idx]}\n")
        
        print(f"   💾 {split_name}: {lr_split_file}, {hr_split_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear pares consistentes y pre-shuffleados de imágenes LR-HR")
    parser.add_argument("--hr_meta", required=True, help="Archivo meta_info de imágenes de alta resolución")
    parser.add_argument("--lr_meta", required=True, help="Archivo meta_info de imágenes de baja resolución")
    parser.add_argument("--output_dir", default="paired_meta", help="Directorio de salida para los archivos meta_info emparejados")
    parser.add_argument("--sample_size", type=int, default=None, help="Tamaño de la muestra (opcional)")
    parser.add_argument("--shuffle_variants", type=int, default=1, help="Número de variantes con diferentes shuffles")
    parser.add_argument("--create_splits", action="store_true", help="Crear splits train/val/test")
    parser.add_argument("--verify", action="store_true", default=True, help="Verificar alineamiento de pares")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    
    args = parser.parse_args()
    
    print("🚀 CREANDO DATASET PRE-SHUFFLEADO OPTIMIZADO")
    print("="*60)
    
    outputs = create_meta_pairs(
        args.hr_meta, 
        args.lr_meta, 
        args.output_dir, 
        args.sample_size,
        args.shuffle_variants,
        args.verify,
        args.seed
    )
    
    # Crear splits si se solicita
    if args.create_splits and outputs:
        lr_file, hr_file = outputs[0]  # Usar primera variante para splits
        create_dataset_splits(lr_file, hr_file, args.output_dir, seed=args.seed)
    
    print(f"\n✅ COMPLETADO")
    print(f"📊 Para usar en entrenamiento:")
    print(f"   - Usar buffer_size=1 en TensorFlow")
    print(f"   - Dataset ya está perfectamente mezclado")
    print(f"   - Pares siempre alineados")