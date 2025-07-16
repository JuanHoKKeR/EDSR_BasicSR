#!/usr/bin/env python3
"""
Script para probar la corrección de las métricas de microscopía.
Verifica que las métricas calculen valores razonables.
"""

import torch
import numpy as np
import cv2
from basicsr.metrics.microscopy_metrics import (
    calculate_psnr_micro, 
    calculate_ssim_micro, 
    calculate_msssim, 
    calculate_mse
)

def test_metrics():
    """Prueba las métricas con diferentes casos"""
    
    print("=" * 60)
    print("PRUEBA DE CORRECCIÓN DE MÉTRICAS DE MICROSCOPÍA")
    print("=" * 60)
    
    # Crear una imagen de prueba
    height, width = 256, 256
    
    # Test 1: Imagen idéntica (debería dar métricas perfectas)
    print("\n1. Prueba con imágenes idénticas:")
    img1 = torch.rand(3, height, width)  # Tensor en [0, 1]
    img2 = img1.clone()
    
    psnr = calculate_psnr_micro(img1, img2)
    ssim = calculate_ssim_micro(img1, img2)
    msssim = calculate_msssim(img1, img2)
    mse = calculate_mse(img1, img2)
    
    print(f"   PSNR: {psnr:.4f} dB (esperado: muy alto ~100)")
    print(f"   SSIM: {ssim:.4f} (esperado: 1.0)")
    print(f"   MS-SSIM: {msssim:.4f} (esperado: 1.0)")
    print(f"   MSE: {mse:.6f} (esperado: ~0)")
    
    # Test 2: Imagen con ruido gaussiano leve
    print("\n2. Prueba con ruido gaussiano leve (sigma=0.01):")
    noise = torch.randn_like(img1) * 0.01
    img2_noisy = torch.clamp(img1 + noise, 0, 1)
    
    psnr = calculate_psnr_micro(img1, img2_noisy)
    ssim = calculate_ssim_micro(img1, img2_noisy)
    msssim = calculate_msssim(img1, img2_noisy)
    mse = calculate_mse(img1, img2_noisy)
    
    print(f"   PSNR: {psnr:.4f} dB (esperado: 35-45 dB)")
    print(f"   SSIM: {ssim:.4f} (esperado: 0.95-0.99)")
    print(f"   MS-SSIM: {msssim:.4f} (esperado: 0.95-0.99)")
    print(f"   MSE: {mse:.6f} (esperado: 0.0001-0.001)")
    
    # Test 3: Imagen con ruido moderado
    print("\n3. Prueba con ruido gaussiano moderado (sigma=0.05):")
    noise = torch.randn_like(img1) * 0.05
    img2_noisy = torch.clamp(img1 + noise, 0, 1)
    
    psnr = calculate_psnr_micro(img1, img2_noisy)
    ssim = calculate_ssim_micro(img1, img2_noisy)
    msssim = calculate_msssim(img1, img2_noisy)
    mse = calculate_mse(img1, img2_noisy)
    
    print(f"   PSNR: {psnr:.4f} dB (esperado: 25-35 dB)")
    print(f"   SSIM: {ssim:.4f} (esperado: 0.85-0.95)")
    print(f"   MS-SSIM: {msssim:.4f} (esperado: 0.85-0.95)")
    print(f"   MSE: {mse:.6f} (esperado: 0.001-0.01)")
    
    # Test 4: Imagen completamente diferente
    print("\n4. Prueba con imagen completamente diferente:")
    img2_diff = torch.rand(3, height, width)
    
    psnr = calculate_psnr_micro(img1, img2_diff)
    ssim = calculate_ssim_micro(img1, img2_diff)
    msssim = calculate_msssim(img1, img2_diff)
    mse = calculate_mse(img1, img2_diff)
    
    print(f"   PSNR: {psnr:.4f} dB (esperado: 8-15 dB)")
    print(f"   SSIM: {ssim:.4f} (esperado: 0.1-0.3)")
    print(f"   MS-SSIM: {msssim:.4f} (esperado: 0.1-0.3)")
    print(f"   MSE: {mse:.6f} (esperado: 0.05-0.1)")
    
    # Test 5: Con arrays de numpy (formato [0, 255])
    print("\n5. Prueba con arrays numpy [0, 255]:")
    img1_np = (img1.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img2_np = (img2_noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    psnr = calculate_psnr_micro(img1_np, img2_np)
    ssim = calculate_ssim_micro(img1_np, img2_np)
    msssim = calculate_msssim(img1_np, img2_np)
    mse = calculate_mse(img1_np, img2_np)
    
    print(f"   PSNR: {psnr:.4f} dB")
    print(f"   SSIM: {ssim:.4f}")
    print(f"   MS-SSIM: {msssim:.4f}")
    print(f"   MSE: {mse:.6f}")
    
    # Test 6: Con crop_border
    print("\n6. Prueba con crop_border=4:")
    psnr = calculate_psnr_micro(img1, img2_noisy, crop_border=4)
    ssim = calculate_ssim_micro(img1, img2_noisy, crop_border=4)
    msssim = calculate_msssim(img1, img2_noisy, crop_border=4)
    mse = calculate_mse(img1, img2_noisy, crop_border=4)
    
    print(f"   PSNR: {psnr:.4f} dB")
    print(f"   SSIM: {ssim:.4f}")
    print(f"   MS-SSIM: {msssim:.4f}")
    print(f"   MSE: {mse:.6f}")
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA")
    print("=" * 60)
    print("\nSi los valores están en los rangos esperados, las métricas")
    print("han sido corregidas exitosamente.")
    print("\nLos valores extremadamente altos de PSNR (>60 dB) y SSIM (>0.99)")
    print("que aparecían antes eran indicativos de problemas en el cálculo.")

if __name__ == "__main__":
    test_metrics()
