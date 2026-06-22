import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertConfig, BertModel, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import math


class GeometricMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(GeometricMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output, attention_weights


class GeometricTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(GeometricTransformerBlock, self).__init__()
        
        self.attention = GeometricMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class GeometricTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                 max_seq_length=1024, dropout=0.1, num_classes=None):
        super(GeometricTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            GeometricTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        if num_classes:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size)
    
    def _create_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        if x.device != self.positional_encoding.device:
            self.positional_encoding = self.positional_encoding.to(x.device)
        
        x = x + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        if self.num_classes:
            x = x.mean(dim=1)
            x = self.classifier(x)
        else:
            x = self.lm_head(x)
        
        return x, attention_weights


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


class TransformerTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs, _ = self.model(input_ids, attention_mask)
            
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs, _ = self.model(input_ids, attention_mask)
                
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1), 
                    ignore_index=-100
                )
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def train(self, train_loader, val_loader, epochs=20):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    def plot_training(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(self.train_losses, label='Train Loss')
        ax.plot(self.val_losses, label='Validation Loss')
        ax.set_title('Training Progress')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        plt.show()


def create_simple_dataset():
    texts = [
        "El aprendizaje profundo geométrico es una extensión del aprendizaje profundo.",
        "Los grafos son estructuras matemáticas que representan relaciones entre entidades.",
        "Las redes neuronales convolucionales procesan datos con estructura de grilla.",
        "Los transformers utilizan mecanismos de atención para procesar secuencias.",
        "La geometría diferencial proporciona herramientas para el análisis de variedades.",
        "Los espacios métricos definen nociones de distancia entre puntos.",
        "El álgebra lineal es fundamental para entender las redes neuronales.",
        "La topología estudia propiedades que se preservan bajo deformaciones continuas.",
        "Los operadores de convolución son equivariantes a las traslaciones.",
        "La teoría de grafos espectrales utiliza valores propios de matrices de adyacencia."
    ] * 100
    
    return texts


def train_geometric_transformer_small():
    print("Entrenando Geometric Transformer (pequeño)")
    
    texts = create_simple_dataset()
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = SimpleTextDataset(texts, tokenizer, max_length=64)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = GeometricTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_length=64
    )
    
    trainer = TransformerTrainer(model)
    trainer.train(train_loader, val_loader, epochs=20)
    trainer.plot_training()
    
    return model, trainer


def train_pretrained_transformer():
    print("Entrenando Transformer preentrenado (BERT)")
    
    from transformers import BertForSequenceClassification, BertTokenizer
    from datasets import Dataset
    
    texts = [
        ("El aprendizaje profundo es fascinante", 1),
        ("No me gusta esta película", 0),
        ("Los grafos son estructuras muy útiles", 1),
        ("Este producto es terrible", 0),
        ("La geometría diferencial es hermosa", 1),
        ("Odio los lunes", 0)
    ] * 50
    
    train_texts = [t[0] for t in texts]
    train_labels = [t[1] for t in texts]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    
    class SentimentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    dataset = SentimentDataset(encodings, train_labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')
    
    return model


if __name__ == "__main__":
    print("=== Modelos de Atención y Transformers ===")
    
    transformer_model, transformer_trainer = train_geometric_transformer_small()
    print(f"Geometric Transformer - Loss final: {transformer_trainer.val_losses[-1]:.4f}")
    
    bert_model = train_pretrained_transformer()
    print("BERT fine-tuning completado")