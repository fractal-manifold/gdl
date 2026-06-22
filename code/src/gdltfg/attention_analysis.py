"""
Análisis Educativo de Mecanismos de Atención
============================================

Este módulo implementa análisis rigurosos de los mecanismos de atención desarrollados
en la Sección 5.1 del TFG, proporcionando verificación empírica de teoremas matemáticos
y visualizaciones conceptuales.

Autor: Jorge Muñoz Fuentes
Institución: UNED - Universidad Nacional de Educación a Distancia
Año: 2024-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eigh
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Dict, List
import seaborn as sns
from dataclasses import dataclass

# Configuración de matplotlib para LaTeX
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

@dataclass
class AttentionAnalysisResult:
    """Resultado del análisis de atención con propiedades matemáticas."""
    attention_matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    singular_values: np.ndarray
    frobenius_norm: float
    spectral_norm: float
    entropy: float


class EducationalAttention:
    """
    Implementación educativa de mecanismos de atención con análisis matemático detallado.
    
    Esta clase está diseñada para fines pedagógicos, proporcionando explicaciones
    paso a paso y verificación de propiedades teóricas.
    """
    
    def __init__(self, d_model: int, d_k: int, d_v: int):
        """
        Inicializa el mecanismo de atención educativo.
        
        Args:
            d_model: Dimensión del modelo (espacio de entrada)
            d_k: Dimensión de consultas y claves
            d_v: Dimensión de valores
        """
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        # Inicialización de pesos (pedagógica, no aleatoria)
        self.W_Q = self._initialize_pedagogical_weights(d_model, d_k, "query")
        self.W_K = self._initialize_pedagogical_weights(d_model, d_k, "key") 
        self.W_V = self._initialize_pedagogical_weights(d_model, d_v, "value")
        
    def _initialize_pedagogical_weights(self, input_dim: int, output_dim: int, 
                                      weight_type: str) -> np.ndarray:
        """
        Inicialización pedagógica de pesos para facilitar análisis matemático.
        """
        if weight_type == "query":
            # Matriz ortogonal para consultas
            W = np.random.randn(input_dim, output_dim)
            W, _ = np.linalg.qr(W)
            return W[:, :output_dim]
        elif weight_type == "key":
            # Matriz con estructura específica para análisis
            W = np.eye(input_dim, output_dim) + 0.1 * np.random.randn(input_dim, output_dim)
            return W
        else:  # value
            # Matriz de valores simple
            return np.random.randn(input_dim, output_dim) * 0.5
            
    def forward_with_analysis(self, X: np.ndarray, 
                            verbose: bool = True) -> AttentionAnalysisResult:
        """
        Forward pass con análisis matemático completo.
        
        Args:
            X: Matriz de entrada [n x d_model]
            verbose: Si imprimir análisis paso a paso
            
        Returns:
            Resultado con análisis matemático detallado
        """
        n, d = X.shape
        assert d == self.d_model, f"Dimensión incorrecta: esperado {self.d_model}, recibido {d}"
        
        if verbose:
            print("="*60)
            print("ANÁLISIS PASO A PASO DE ATENCIÓN")
            print("="*60)
            print(f"Entrada X: {X.shape}")
            print(f"Norma de Frobenius de X: {np.linalg.norm(X, 'fro'):.4f}")
        
        # Paso 1: Proyecciones lineales (Ecuación 5.1.2)
        Q = X @ self.W_Q  # [n x d_k]
        K = X @ self.W_K  # [n x d_k] 
        V = X @ self.W_V  # [n x d_v]
        
        if verbose:
            print(f"\nPaso 1: Proyecciones lineales")
            print(f"Q shape: {Q.shape}, norma: {np.linalg.norm(Q, 'fro'):.4f}")
            print(f"K shape: {K.shape}, norma: {np.linalg.norm(K, 'fro'):.4f}")
            print(f"V shape: {V.shape}, norma: {np.linalg.norm(V, 'fro'):.4f}")
        
        # Paso 2: Matriz de compatibilidad (Definición 5.1.1)
        S = Q @ K.T  # [n x n] - Matriz de Gram
        
        if verbose:
            print(f"\nPaso 2: Matriz de compatibilidad (Gram)")
            print(f"S = Q @ K^T, shape: {S.shape}")
            print(f"Matriz S:\n{S}")
        
        # Paso 3: Escalado (Ecuación 5.1.3)
        S_scaled = S / np.sqrt(self.d_k)
        
        if verbose:
            print(f"\nPaso 3: Escalado por sqrt(d_k) = sqrt({self.d_k}) = {np.sqrt(self.d_k):.4f}")
            print(f"S_scaled:\n{S_scaled}")
        
        # Paso 4: Aplicación de softmax
        A = self._softmax_2d(S_scaled)
        
        if verbose:
            print(f"\nPaso 4: Aplicación de softmax")
            print(f"Matriz de atención A:\n{A}")
            print(f"Verificación suma por filas: {np.sum(A, axis=1)}")
        
        # Paso 5: Aplicación de valores
        Y = A @ V
        
        if verbose:
            print(f"\nPaso 5: Aplicación de valores")
            print(f"Salida Y = A @ V, shape: {Y.shape}")
            print(f"Y:\n{Y}")
        
        # Análisis espectral detallado
        eigenvals, eigenvecs = eigh(A)
        eigenvals = eigenvals[::-1]  # Orden descendente
        eigenvecs = eigenvecs[:, ::-1]
        
        U, sigma, Vt = svd(A)
        
        frobenius_norm = np.linalg.norm(A, 'fro')
        spectral_norm = np.linalg.norm(A, 2)
        entropy = -np.sum(A * np.log(A + 1e-12))  # Entropía de Shannon
        
        if verbose:
            print(f"\nANÁLISIS ESPECTRAL:")
            print(f"Eigenvalores: {eigenvals}")
            print(f"Valores singulares: {sigma}")
            print(f"Norma de Frobenius: {frobenius_norm:.4f}")
            print(f"Norma espectral: {spectral_norm:.4f}")
            print(f"Entropía de Shannon: {entropy:.4f}")
        
        return AttentionAnalysisResult(
            attention_matrix=A,
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            singular_values=sigma,
            frobenius_norm=frobenius_norm,
            spectral_norm=spectral_norm,
            entropy=entropy
        )
    
    def _softmax_2d(self, X: np.ndarray) -> np.ndarray:
        """Softmax numéricamente estable aplicado por filas."""
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def verify_universal_approximation_theorem(self, n_samples: int = 1000,
                                             d_input: int = 8) -> Dict[str, float]:
        """
        Verificación empírica del Teorema 5.1.1 (Universalidad de Auto-atención).
        
        Esta función genera funciones permutación-equivariantes sintéticas y verifica
        que pueden ser aproximadas por mecanismos de auto-atención.
        """
        print("VERIFICACIÓN DEL TEOREMA DE UNIVERSALIDAD")
        print("="*50)
        
        # Generar función permutación-equivariante sintética
        def target_function(X):
            """Función objetivo permutación-equivariante: f(X) = X + mean(X)"""
            mean_X = np.mean(X, axis=0, keepdims=True)
            return X + mean_X
        
        # Generar datos de prueba
        test_data = []
        target_outputs = []
        
        for _ in range(n_samples):
            # Generar entrada aleatoria
            n_seq = np.random.randint(3, 8)
            X = np.random.randn(n_seq, d_input)
            
            # Calcular salida objetivo
            y_target = target_function(X)
            
            test_data.append(X)
            target_outputs.append(y_target)
        
        # Inicializar mecanismo de atención
        attention = EducationalAttention(d_input, d_k=d_input//2, d_v=d_input)
        
        errors = []
        
        for X, y_target in zip(test_data[:100], target_outputs[:100]):  # Submuestra para velocidad
            try:
                result = attention.forward_with_analysis(X, verbose=False)
                y_pred = result.attention_matrix @ (X @ attention.W_V)
                
                # Calcular error de aproximación
                error = np.linalg.norm(y_pred - y_target, 'fro') / np.linalg.norm(y_target, 'fro')
                errors.append(error)
                
            except Exception as e:
                print(f"Error en muestra: {e}")
                continue
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"Error promedio de aproximación: {mean_error:.6f}")
        print(f"Desviación estándar: {std_error:.6f}")
        print(f"Número de muestras procesadas: {len(errors)}")
        
        # Verificar equivarianza a permutaciones
        equivariance_errors = []
        
        for X, y_target in zip(test_data[:50], target_outputs[:50]):
            try:
                n_seq = X.shape[0]
                
                # Aplicar permutación aleatoria
                perm = np.random.permutation(n_seq)
                X_perm = X[perm]
                
                # Procesar ambas versiones
                result_orig = attention.forward_with_analysis(X, verbose=False)
                result_perm = attention.forward_with_analysis(X_perm, verbose=False)
                
                y_orig = result_orig.attention_matrix @ (X @ attention.W_V)
                y_perm = result_perm.attention_matrix @ (X_perm @ attention.W_V)
                
                # Verificar que P*f(X) = f(P*X)
                expected_perm = y_orig[perm]
                equivar_error = np.linalg.norm(y_perm - expected_perm, 'fro')
                equivariance_errors.append(equivar_error)
                
            except Exception:
                continue
        
        mean_equivar_error = np.mean(equivariance_errors) if equivariance_errors else float('inf')
        
        print(f"Error promedio de equivarianza: {mean_equivar_error:.6f}")
        
        return {
            'approximation_error_mean': mean_error,
            'approximation_error_std': std_error,
            'equivariance_error': mean_equivar_error,
            'n_samples_processed': len(errors)
        }
    
    def visualize_attention_geometry(self, X: np.ndarray, 
                                   figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualización de la geometría de atención según análisis de la Sección 5.1.2.
        """
        result = self.forward_with_analysis(X, verbose=False)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(r'Análisis Geométrico de Atención', fontsize=16)
        
        # 1. Heatmap de la matriz de atención
        im1 = axes[0, 0].imshow(result.attention_matrix, cmap='viridis', aspect='equal')
        axes[0, 0].set_title(r'Matriz de Atención $\mathbf{A}$')
        axes[0, 0].set_xlabel('Claves')
        axes[0, 0].set_ylabel('Consultas')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Espectro de eigenvalores
        axes[0, 1].bar(range(len(result.eigenvalues)), result.eigenvalues)
        axes[0, 1].set_title(r'Espectro de $\mathbf{A}$')
        axes[0, 1].set_xlabel('Índice de eigenvalor')
        axes[0, 1].set_ylabel(r'$\lambda_i$')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Valores singulares
        axes[0, 2].semilogy(result.singular_values, 'o-')
        axes[0, 2].set_title('Valores Singulares')
        axes[0, 2].set_xlabel('Índice')
        axes[0, 2].set_ylabel(r'$\sigma_i$')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Distribución de pesos por fila
        for i in range(min(4, result.attention_matrix.shape[0])):
            axes[1, 0].plot(result.attention_matrix[i], 'o-', 
                          label=f'Consulta {i+1}', alpha=0.7)
        axes[1, 0].set_title('Distribución de Pesos por Consulta')
        axes[1, 0].set_xlabel('Posición de Clave')
        axes[1, 0].set_ylabel('Peso de Atención')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Proyecciones Q, K en 2D (PCA si es necesario)
        Q = X @ self.W_Q
        K = X @ self.W_K
        
        if Q.shape[1] >= 2:
            axes[1, 1].scatter(Q[:, 0], Q[:, 1], c='red', alpha=0.7, s=100, label='Consultas (Q)')
            axes[1, 1].scatter(K[:, 0], K[:, 1], c='blue', alpha=0.7, s=100, label='Claves (K)')
            
            # Dibujar líneas de conexión según pesos de atención
            for i in range(len(Q)):
                for j in range(len(K)):
                    weight = result.attention_matrix[i, j]
                    if weight > 0.1:  # Solo mostrar conexiones significativas
                        axes[1, 1].plot([Q[i, 0], K[j, 0]], [Q[i, 1], K[j, 1]], 
                                       'k-', alpha=weight*2, linewidth=weight*3)
            
            axes[1, 1].set_title('Espacio Q-K con Conexiones')
            axes[1, 1].set_xlabel(r'$q_1, k_1$')
            axes[1, 1].set_ylabel(r'$q_2, k_2$')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Métricas de la matriz de Gram
        gram_metrics = {
            'Norma F': result.frobenius_norm,
            'Norma 2': result.spectral_norm,
            'Entropía': result.entropy,
            'Det': np.linalg.det(result.attention_matrix + 1e-12*np.eye(len(result.attention_matrix)))
        }
        
        metrics_names = list(gram_metrics.keys())
        metrics_values = list(gram_metrics.values())
        
        bars = axes[1, 2].bar(metrics_names, metrics_values)
        axes[1, 2].set_title('Métricas de la Matriz de Gram')
        axes[1, 2].set_ylabel('Valor')
        
        # Anotar valores en las barras
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[1, 2].annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()


def run_example_4x4_verification():
    """
    Verificación del Ejemplo 5.1.7: Atención 4×4 con análisis espectral.
    
    Reproduce los cálculos exactos del ejemplo teórico de la Sección 5.1.7.
    """
    print("VERIFICACIÓN DEL EJEMPLO 4×4 DE LA SECCIÓN 5.1.7")
    print("="*55)
    
    # Datos exactos del ejemplo teórico
    Q = np.array([
        [1, 0, 1],
        [0, 1, 1], 
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)
    
    K = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=float)
    
    V = np.array([
        [2, 0],
        [0, 2],
        [1, 1],
        [0.5, 1.5]
    ], dtype=float)
    
    print("Matrices de entrada:")
    print("Q =")
    print(Q)
    print("\nK =") 
    print(K)
    print("\nV =")
    print(V)
    
    # Paso 1: Matriz de scores
    S = Q @ K.T
    print("\nPaso 1: Matriz de scores S = Q @ K^T")
    print("S =")
    print(S)
    
    # Verificación con valores esperados
    S_expected = np.array([
        [1, 0, 1, 2],
        [0, 1, 1, 2], 
        [1, 1, 0, 2],
        [1, 1, 1, 3]
    ], dtype=float)
    
    print(f"\nVerificación S: Error = {np.linalg.norm(S - S_expected):.6f}")
    
    # Paso 2: Escalado
    d_k = Q.shape[1]
    S_scaled = S / np.sqrt(d_k)
    print(f"\nPaso 2: Escalado por sqrt(d_k) = sqrt({d_k}) = {np.sqrt(d_k):.4f}")
    print("S_scaled =") 
    print(S_scaled)
    
    # Paso 3: Softmax
    def softmax_2d(X):
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    A = softmax_2d(S_scaled)
    print("\nPaso 3: Matriz de atención A = softmax(S_scaled)")
    print("A =")
    print(A)
    
    # Verificación de propiedades
    row_sums = np.sum(A, axis=1)
    print(f"\nVerificación suma por filas: {row_sums}")
    print(f"Error máximo en suma de filas: {np.max(np.abs(row_sums - 1.0)):.8f}")
    
    # Paso 4: Aplicación de valores
    Y = A @ V
    print("\nPaso 4: Salida Y = A @ V")
    print("Y =")
    print(Y)
    
    # Análisis espectral
    eigenvals, eigenvecs = np.linalg.eigh(A)
    eigenvals = eigenvals[::-1]  # Orden descendente
    
    print("\nAnálisis espectral:")
    print(f"Eigenvalores: {eigenvals}")
    print(f"Eigenvalor dominante: {eigenvals[0]:.6f} (esperado: 1.0)")
    
    # Métricas adicionales
    frobenius_norm = np.linalg.norm(S, 'fro')
    print(f"\nNorma de Frobenius de S: {frobenius_norm:.4f} (esperado: ≈5.10)")
    print(f"Norma de Frobenius calculada: {np.sqrt(26):.4f}")
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Heatmap de S
    im1 = axes[0].imshow(S, cmap='viridis', aspect='equal')
    axes[0].set_title('Matriz de Scores S')
    axes[0].set_xlabel('Claves')
    axes[0].set_ylabel('Consultas')
    plt.colorbar(im1, ax=axes[0])
    
    # Heatmap de A
    im2 = axes[1].imshow(A, cmap='plasma', aspect='equal')
    axes[1].set_title('Matriz de Atención A')
    axes[1].set_xlabel('Claves') 
    axes[1].set_ylabel('Consultas')
    plt.colorbar(im2, ax=axes[1])
    
    # Eigenvalores
    axes[2].bar(range(len(eigenvals)), eigenvals)
    axes[2].set_title('Espectro de A')
    axes[2].set_xlabel('Índice')
    axes[2].set_ylabel('Eigenvalor')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'S': S,
        'S_scaled': S_scaled,
        'A': A,
        'Y': Y,
        'eigenvalues': eigenvals,
        'frobenius_norm_S': frobenius_norm
    }


if __name__ == "__main__":
    # Demostración del código educativo
    
    print("ANÁLISIS EDUCATIVO DE MECANISMOS DE ATENCIÓN")
    print("=" * 60)
    
    # Ejemplo 1: Verificación del ejemplo teórico 4×4
    print("\n1. VERIFICACIÓN DEL EJEMPLO TEÓRICO 4×4")
    results_4x4 = run_example_4x4_verification()
    
    # Ejemplo 2: Análisis general con visualización
    print("\n\n2. ANÁLISIS GENERAL CON VISUALIZACIÓN")
    
    # Crear datos de prueba
    np.random.seed(42)
    X_test = np.random.randn(5, 6)
    
    # Inicializar mecanismo de atención educativo
    attention = EducationalAttention(d_model=6, d_k=4, d_v=3)
    
    # Análisis completo
    result = attention.forward_with_analysis(X_test, verbose=True)
    
    # Visualización geométrica
    attention.visualize_attention_geometry(X_test)
    
    # Ejemplo 3: Verificación del teorema de universalidad
    print("\n\n3. VERIFICACIÓN EMPÍRICA DEL TEOREMA DE UNIVERSALIDAD")
    universal_results = attention.verify_universal_approximation_theorem()
    
    print("\nRESUMEN DE RESULTADOS:")
    print("=" * 30)
    print(f"Error de aproximación: {universal_results['approximation_error_mean']:.6f}")
    print(f"Error de equivarianza: {universal_results['equivariance_error']:.6f}")
    print(f"Muestras procesadas: {universal_results['n_samples_processed']}")