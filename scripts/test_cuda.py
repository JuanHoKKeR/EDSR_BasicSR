#!/usr/bin/env python3
"""
Script rápido para verificar CUDA y GPU
"""

import torch
import time

def test_cuda():
    print("=== TEST CUDA Y GPU ===\n")
    
    # 1. Verificar si CUDA está disponible
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA no está disponible!")
        return False
    
    # 2. Información de la GPU
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    print(f"GPU actual: {torch.cuda.current_device()}")
    print(f"Nombre GPU: {torch.cuda.get_device_name()}")
    print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 3. Test básico de operaciones
    print("\n--- Test de operaciones básicas ---")
    
    # Crear tensores en GPU
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    # Operación simple
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()  # Esperar a que termine
    end_time = time.time()
    
    print(f"Multiplicación de matrices 1000x1000: {end_time - start_time:.4f} segundos")
    print(f"Resultado correcto: {z.shape == (1000, 1000)}")
    
    # 4. Test de memoria
    print("\n--- Test de memoria ---")
    print(f"Memoria usada: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Memoria reservada: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # 5. Test de precisión
    print("\n--- Test de precisión ---")
    a = torch.tensor([1.0, 2.0, 3.0]).to(device)
    b = torch.tensor([4.0, 5.0, 6.0]).to(device)
    c = a + b
    print(f"Suma: {a} + {b} = {c}")
    print(f"Resultado correcto: {torch.allclose(c, torch.tensor([5.0, 7.0, 9.0]).to(device))}")
    
    # 6. Limpiar memoria
    del x, y, z, a, b, c
    torch.cuda.empty_cache()
    
    print("\n✅ CUDA y GPU funcionando correctamente!")
    return True

if __name__ == "__main__":
    test_cuda() 