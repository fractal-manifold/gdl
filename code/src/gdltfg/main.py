#!/usr/bin/env python3
"""
Script principal para entrenar y ejecutar modelos de Geometric Deep Learning.
Incluye modelos de grafos (GNN), convoluciones (CNN) y atención (Transformer).
"""

import argparse
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

from models.gnn import train_gcn_model, train_gat_model
from models.cnn import train_geometric_cnn, train_geometric_resnet
from models.transformer import train_geometric_transformer_small, train_pretrained_transformer


def print_banner():
    print("=" * 60)
    print("  GEOMETRIC DEEP LEARNING - MODELOS DE DEMOSTRACIÓN")
    print("=" * 60)
    print("  Implementaciones de GNN, CNN y Transformers")
    print("  para la tesis de Grado en Matemáticas - UNED")
    print("=" * 60)


def check_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ Usando CPU")
    
    return device


def run_gnn_models():
    print("\n" + "="*50)
    print("ENTRENANDO MODELOS DE REDES DE GRAFOS (GNN)")
    print("="*50)
    
    try:
        print("\n1. Graph Convolutional Network (GCN)")
        print("-" * 40)
        gcn_model, gcn_trainer = train_gcn_model()
        print(f"✓ GCN entrenado. Precisión final: {gcn_trainer.val_accuracies[-1]:.4f}")
        
        print("\n2. Graph Attention Network (GAT)")
        print("-" * 40)
        gat_model, gat_trainer = train_gat_model()
        print(f"✓ GAT entrenado. Precisión final: {gat_trainer.val_accuracies[-1]:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error en modelos GNN: {e}")
        return False


def run_cnn_models():
    print("\n" + "="*50)
    print("ENTRENANDO MODELOS CONVOLUCIONALES (CNN)")
    print("="*50)
    
    try:
        print("\n1. Geometric CNN")
        print("-" * 40)
        cnn_model, cnn_trainer = train_geometric_cnn()
        print(f"✓ Geometric CNN entrenado. Precisión final: {cnn_trainer.val_accuracies[-1]:.2f}%")
        
        print("\n2. Geometric ResNet")
        print("-" * 40)
        resnet_model, resnet_trainer = train_geometric_resnet()
        print(f"✓ Geometric ResNet entrenado. Precisión final: {resnet_trainer.val_accuracies[-1]:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ Error en modelos CNN: {e}")
        return False


def run_transformer_models():
    print("\n" + "="*50)
    print("ENTRENANDO MODELOS DE ATENCIÓN (TRANSFORMERS)")
    print("="*50)
    
    try:
        print("\n1. Geometric Transformer (desde cero)")
        print("-" * 40)
        transformer_model, transformer_trainer = train_geometric_transformer_small()
        print(f"✓ Geometric Transformer entrenado. Loss final: {transformer_trainer.val_losses[-1]:.4f}")
        
        print("\n2. BERT Fine-tuning")
        print("-" * 40)
        bert_model = train_pretrained_transformer()
        print("✓ BERT fine-tuning completado")
        
        return True
    except Exception as e:
        print(f"✗ Error en modelos Transformer: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar modelos de Geometric Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --all                 # Entrenar todos los modelos
  python main.py --gnn                 # Solo modelos de grafos
  python main.py --cnn                 # Solo modelos convolucionales
  python main.py --transformer         # Solo modelos de atención
  python main.py --gnn --cnn           # Grafos y convoluciones
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Entrenar todos los modelos')
    parser.add_argument('--gnn', action='store_true',
                       help='Entrenar modelos de redes de grafos (GCN, GAT)')
    parser.add_argument('--cnn', action='store_true',
                       help='Entrenar modelos convolucionales (CNN, ResNet)')
    parser.add_argument('--transformer', action='store_true',
                       help='Entrenar modelos de atención (Transformer, BERT)')
    parser.add_argument('--no-plots', action='store_true',
                       help='No mostrar gráficos de entrenamiento')
    
    args = parser.parse_args()
    
    if not any([args.all, args.gnn, args.cnn, args.transformer]):
        parser.print_help()
        return
    
    print_banner()
    device = check_device()
    
    results = {
        'gnn': False,
        'cnn': False, 
        'transformer': False
    }
    
    if args.all or args.gnn:
        results['gnn'] = run_gnn_models()
    
    if args.all or args.cnn:
        results['cnn'] = run_cnn_models()
    
    if args.all or args.transformer:
        results['transformer'] = run_transformer_models()
    
    print("\n" + "="*60)
    print("RESUMEN DE ENTRENAMIENTO")
    print("="*60)
    
    if results['gnn']:
        print("✓ Modelos de Grafos (GNN): Completado")
    elif args.gnn or args.all:
        print("✗ Modelos de Grafos (GNN): Error")
    
    if results['cnn']:
        print("✓ Modelos Convolucionales (CNN): Completado")
    elif args.cnn or args.all:
        print("✗ Modelos Convolucionales (CNN): Error")
    
    if results['transformer']:
        print("✓ Modelos de Atención (Transformer): Completado")
    elif args.transformer or args.all:
        print("✗ Modelos de Atención (Transformer): Error")
    
    success_count = sum(results.values())
    total_requested = sum([args.gnn or args.all, args.cnn or args.all, args.transformer or args.all])
    
    print(f"\nModelos completados exitosamente: {success_count}/{total_requested}")
    
    if success_count == total_requested:
        print("\n🎉 ¡Todos los modelos se entrenaron correctamente!")
        return 0
    else:
        print("\n⚠️  Algunos modelos presentaron errores.")
        return 1


if __name__ == "__main__":
    sys.exit(main())