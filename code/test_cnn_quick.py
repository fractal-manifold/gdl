#!/usr/bin/env python3
"""Script de prueba rápida para modelos CNN con datasets sintéticos"""

import sys
import os
sys.path.append('src')

from gdltfg.models.cnn import train_geometric_cnn

if __name__ == "__main__":
    print("=== Prueba Rápida de CNN con Dataset Sintético ===")
    
    # Desactivar matplotlib plotting para evitar warnings
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        print("\n1. Entrenando Geometric CNN (5 epochs)")
        cnn_model, cnn_trainer = train_geometric_cnn(use_synthetic=True)
        print(f"✓ Geometric CNN entrenado. Precisión final: {cnn_trainer.val_accuracies[-1]:.2f}%")
        
        print("\n🎉 ¡Modelo CNN probado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()