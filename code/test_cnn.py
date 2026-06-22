#!/usr/bin/env python3
"""Script de prueba para modelos CNN con datasets sintéticos"""

import sys
sys.path.append('src')

from gdltfg.models.cnn import train_geometric_cnn, train_geometric_resnet

if __name__ == "__main__":
    print("=== Prueba de Modelos CNN con Dataset Sintético ===")
    
    try:
        print("\n1. Entrenando Geometric CNN")
        cnn_model, cnn_trainer = train_geometric_cnn(use_synthetic=True)
        print(f"✓ Geometric CNN entrenado. Precisión final: {cnn_trainer.val_accuracies[-1]:.2f}%")
        
        print("\n2. Entrenando Geometric ResNet")
        resnet_model, resnet_trainer = train_geometric_resnet(use_synthetic=True)
        print(f"✓ Geometric ResNet entrenado. Precisión final: {resnet_trainer.val_accuracies[-1]:.2f}%")
        
        print("\n🎉 ¡Modelos CNN probados exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()