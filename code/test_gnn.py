#!/usr/bin/env python3
"""Script de prueba para modelos GNN con datasets sintéticos"""

import sys
sys.path.append('src')

from gdltfg.models.gnn import train_gcn_model, train_gat_model

if __name__ == "__main__":
    print("=== Prueba de Modelos GNN con Dataset Sintético ===")
    
    try:
        print("\n1. Entrenando Graph Convolutional Network (GCN)")
        gcn_model, gcn_trainer = train_gcn_model(use_synthetic=True)
        print(f"✓ GCN entrenado. Precisión final: {gcn_trainer.val_accuracies[-1]:.4f}")
        
        print("\n2. Entrenando Graph Attention Network (GAT)")
        gat_model, gat_trainer = train_gat_model(use_synthetic=True)
        print(f"✓ GAT entrenado. Precisión final: {gat_trainer.val_accuracies[-1]:.4f}")
        
        print("\n🎉 ¡Modelos GNN probados exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()