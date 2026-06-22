import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset, TensorDataset
import random
import math


class SyntheticGraphDataset(InMemoryDataset):
    def __init__(self, num_graphs=1000, num_nodes_range=(10, 50), num_features=10, num_classes=3):
        self.num_graphs = num_graphs
        self.num_nodes_range = num_nodes_range
        self._num_features = num_features
        self._num_classes = num_classes
        super().__init__(root=None)
        self.data, self.slices = self.collate(self._generate_graphs())
    
    @property
    def num_node_features(self):
        return self._num_features
    
    @property
    def num_classes(self):
        return self._num_classes
    
    def _generate_graphs(self):
        graphs = []
        for i in range(self.num_graphs):
            num_nodes = random.randint(*self.num_nodes_range)
            
            # Generar características de nodos
            x = torch.randn(num_nodes, self._num_features)
            
            # Generar grafo aleatorio con probabilidad de conexión basada en características
            edge_list = []
            for src in range(num_nodes):
                for dst in range(src + 1, num_nodes):
                    # Probabilidad de conexión basada en similitud de características
                    similarity = F.cosine_similarity(x[src], x[dst], dim=0)
                    prob = torch.sigmoid(similarity * 2).item()  # Escalar para mejor distribución
                    
                    if random.random() < prob:
                        edge_list.extend([[src, dst], [dst, src]])  # Grafo no dirigido
            
            if not edge_list:
                # Asegurar al menos una arista
                edge_list = [[0, 1], [1, 0]] if num_nodes > 1 else [[0, 0]]
            
            edge_index = torch.tensor(edge_list).t().contiguous()
            
            # Etiqueta basada en propiedades del grafo
            avg_degree = edge_index.size(1) / num_nodes
            node_feature_sum = x.sum().item()
            
            # Clasificación simple basada en características
            if avg_degree > 4 and node_feature_sum > 0:
                y = 0  # Grafo denso y características positivas
            elif avg_degree <= 2:
                y = 1  # Grafo disperso
            else:
                y = 2  # Intermedio
            
            y = torch.tensor([y], dtype=torch.long)
            
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        
        return graphs


class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=10000, image_size=32, num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = []
        self.targets = []
        self._generate_images()
    
    def _generate_images(self):
        print(f"Generando {self.num_samples} imágenes sintéticas...")
        
        for i in range(self.num_samples):
            # Generar diferentes tipos de patrones geométricos
            pattern_type = i % self.num_classes
            
            # Crear imagen base
            img = torch.zeros(3, self.image_size, self.image_size)
            
            if pattern_type == 0:  # Círculos
                center_x, center_y = np.random.randint(8, 24, 2)
                radius = np.random.randint(5, 12)
                color = torch.rand(3)
                
                y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                
                for c in range(3):
                    img[c][mask] = color[c]
            
            elif pattern_type == 1:  # Cuadrados
                size = np.random.randint(8, 16)
                x1 = np.random.randint(0, self.image_size - size)
                y1 = np.random.randint(0, self.image_size - size)
                color = torch.rand(3)
                
                for c in range(3):
                    img[c, y1:y1+size, x1:x1+size] = color[c]
            
            elif pattern_type == 2:  # Líneas diagonales
                color = torch.rand(3)
                for i in range(0, self.image_size, 2):
                    if i < self.image_size:
                        for c in range(3):
                            img[c, i, :] = color[c] * 0.5
                            if i+1 < self.image_size:
                                img[c, :, i+1] = color[c] * 0.3
            
            elif pattern_type == 3:  # Gradientes
                color1 = torch.rand(3)
                color2 = torch.rand(3)
                
                for i in range(self.image_size):
                    alpha = i / self.image_size
                    color = color1 * (1 - alpha) + color2 * alpha
                    for c in range(3):
                        img[c, i, :] = color[c]
            
            elif pattern_type == 4:  # Ruido estructurado
                noise = torch.randn(3, self.image_size, self.image_size) * 0.3
                base_color = torch.rand(3).unsqueeze(-1).unsqueeze(-1)
                img = base_color + noise
                img = torch.clamp(img, 0, 1)
            
            elif pattern_type == 5:  # Anillos concéntricos
                center_x, center_y = self.image_size // 2, self.image_size // 2
                color = torch.rand(3)
                
                y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
                distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                for r in range(3, 15, 3):
                    mask = (distance >= r) & (distance <= r + 1)
                    for c in range(3):
                        img[c][mask] = color[c] * (r / 15)
            
            elif pattern_type == 6:  # Triángulos
                # Triángulo simple
                points = torch.tensor([[16, 8], [8, 24], [24, 24]], dtype=torch.float)
                color = torch.rand(3)
                
                y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
                coords = torch.stack([x, y], dim=-1).float()
                
                # Método simple para detectar puntos dentro del triángulo
                mask = torch.zeros(self.image_size, self.image_size, dtype=torch.bool)
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        if (i >= 8 and i <= 24 and j >= 8 and j <= 24 and 
                            j >= 8 + (i - 8) * 0.5 and j >= 32 - i):
                            mask[i, j] = True
                
                for c in range(3):
                    img[c][mask] = color[c]
            
            elif pattern_type == 7:  # Patrones de tablero
                block_size = 4
                color1 = torch.rand(3)
                color2 = torch.rand(3)
                
                for i in range(0, self.image_size, block_size):
                    for j in range(0, self.image_size, block_size):
                        color = color1 if ((i//block_size) + (j//block_size)) % 2 == 0 else color2
                        end_i = min(i + block_size, self.image_size)
                        end_j = min(j + block_size, self.image_size)
                        
                        for c in range(3):
                            img[c, i:end_i, j:end_j] = color[c]
            
            elif pattern_type == 8:  # Ondas sinusoidales
                color = torch.rand(3)
                frequency = np.random.uniform(0.1, 0.5)
                
                x = torch.arange(self.image_size).float()
                y = torch.arange(self.image_size).float()
                
                for i in range(self.image_size):
                    wave = torch.sin(x * frequency * 2 * math.pi) * 5 + self.image_size // 2
                    mask = torch.abs(y - wave[i]) < 2
                    for c in range(3):
                        img[c, mask, i] = color[c]
            
            else:  # pattern_type == 9: Formas aleatorias
                num_shapes = np.random.randint(3, 8)
                for _ in range(num_shapes):
                    shape_size = np.random.randint(3, 8)
                    x_pos = np.random.randint(0, self.image_size - shape_size)
                    y_pos = np.random.randint(0, self.image_size - shape_size)
                    color = torch.rand(3)
                    
                    for c in range(3):
                        img[c, y_pos:y_pos+shape_size, x_pos:x_pos+shape_size] = color[c]
            
            # Añadir algo de ruido
            img += torch.randn_like(img) * 0.05
            img = torch.clamp(img, 0, 1)
            
            self.data.append(img)
            self.targets.append(pattern_type)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SyntheticTextDataset(Dataset):
    def __init__(self, num_samples=5000, seq_length=64, vocab_size=1000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = []
        self._generate_sequences()
    
    def _generate_sequences(self):
        print(f"Generando {self.num_samples} secuencias de texto sintéticas...")
        
        # Patrones de secuencias diferentes
        patterns = [
            self._arithmetic_sequence,
            self._geometric_sequence,
            self._fibonacci_like,
            self._palindrome_sequence,
            self._random_walk,
            self._repeating_pattern,
            self._alternating_pattern
        ]
        
        for i in range(self.num_samples):
            pattern_fn = patterns[i % len(patterns)]
            sequence = pattern_fn()
            self.data.append(torch.tensor(sequence, dtype=torch.long))
    
    def _arithmetic_sequence(self):
        start = np.random.randint(1, 100)
        step = np.random.randint(1, 10)
        sequence = [(start + i * step) % self.vocab_size for i in range(self.seq_length)]
        return sequence
    
    def _geometric_sequence(self):
        start = np.random.randint(1, 10)
        ratio = np.random.choice([2, 3, 5])
        sequence = [(start * (ratio ** i)) % self.vocab_size for i in range(self.seq_length)]
        return sequence
    
    def _fibonacci_like(self):
        a, b = np.random.randint(1, 10), np.random.randint(1, 10)
        sequence = [a, b]
        for _ in range(self.seq_length - 2):
            next_val = (sequence[-1] + sequence[-2]) % self.vocab_size
            sequence.append(next_val)
        return sequence
    
    def _palindrome_sequence(self):
        half_length = self.seq_length // 2
        half_seq = [np.random.randint(0, self.vocab_size) for _ in range(half_length)]
        if self.seq_length % 2 == 0:
            sequence = half_seq + half_seq[::-1]
        else:
            middle = [np.random.randint(0, self.vocab_size)]
            sequence = half_seq + middle + half_seq[::-1]
        return sequence
    
    def _random_walk(self):
        current = np.random.randint(0, self.vocab_size)
        sequence = [current]
        
        for _ in range(self.seq_length - 1):
            step = np.random.choice([-1, 0, 1])
            current = (current + step) % self.vocab_size
            sequence.append(current)
        
        return sequence
    
    def _repeating_pattern(self):
        pattern_length = np.random.randint(3, 10)
        pattern = [np.random.randint(0, self.vocab_size) for _ in range(pattern_length)]
        
        sequence = []
        for i in range(self.seq_length):
            sequence.append(pattern[i % pattern_length])
        
        return sequence
    
    def _alternating_pattern(self):
        values = [np.random.randint(0, self.vocab_size) for _ in range(3)]
        sequence = []
        
        for i in range(self.seq_length):
            if i % 3 == 0:
                sequence.append(values[0])
            elif i % 3 == 1:
                sequence.append(values[1])
            else:
                sequence.append(values[2])
        
        return sequence
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Para modeling de lenguaje, input y target son la misma secuencia desplazada
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        # Padding si es necesario
        if len(input_ids) < self.seq_length - 1:
            pad_length = self.seq_length - 1 - len(input_ids)
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            labels = F.pad(labels, (0, pad_length), value=-100)  # -100 ignored in loss
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }


def create_synthetic_datasets():
    """Crear todos los datasets sintéticos"""
    print("Creando datasets sintéticos...")
    
    # Dataset de grafos
    graph_dataset = SyntheticGraphDataset(num_graphs=500, num_classes=3)
    
    # Dataset de imágenes  
    image_dataset = SyntheticImageDataset(num_samples=2000, num_classes=10)
    
    # Dataset de texto
    text_dataset = SyntheticTextDataset(num_samples=1000, seq_length=32, vocab_size=500)
    
    print("✓ Datasets sintéticos creados exitosamente")
    
    return graph_dataset, image_dataset, text_dataset


if __name__ == "__main__":
    graph_ds, image_ds, text_ds = create_synthetic_datasets()
    
    print(f"Graph dataset: {len(graph_ds)} grafos")
    print(f"Image dataset: {len(image_ds)} imágenes") 
    print(f"Text dataset: {len(text_ds)} secuencias")
    
    # Mostrar ejemplos
    print("\nEjemplo de grafo:")
    sample_graph = graph_ds[0]
    print(f"  Nodos: {sample_graph.x.shape[0]}, Características: {sample_graph.x.shape[1]}")
    print(f"  Aristas: {sample_graph.edge_index.shape[1]}, Clase: {sample_graph.y.item()}")
    
    print("\nEjemplo de imagen:")
    sample_img, sample_label = image_ds[0]
    print(f"  Forma: {sample_img.shape}, Clase: {sample_label}")
    
    print("\nEjemplo de texto:")
    sample_text = text_ds[0]
    print(f"  Longitud: {len(sample_text['input_ids'])}")
    print(f"  Tokens: {sample_text['input_ids'][:10]}...")