# ==============================================================================
# DEEP FRAUD HUNTER - Framework de Deep Learning para Detección de Fraude
# ==============================================================================
"""
Estado del Arte en Detección de Fraude Financiero usando Deep Learning.

Este módulo implementa 5 algoritmos SOTA:
1. SiameseEntityResolver - Resolución de entidades duplicadas
2. GraphSAGEEncoder - Embeddings de grafo inductivos
3. FraudVAE - Detección de anomalías con Autoencoder Variacional
4. BERTSemanticAnalyzer - Análisis semántico de texto corporativo
5. TemporalFusionModule - Detección de cambios abruptos temporales

Author: FraudHunter AI Team
Version: 2.0.0 (Deep Learning Edition)
Compatible: PyTorch 2.0+, DGL 1.1+, Transformers 4.30+
"""

import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# ==============================================================================
# OPTIONAL IMPORTS (Graceful degradation si no están disponibles)
# ==============================================================================

# PyTorch Core
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch no disponible. Módulos de Deep Learning deshabilitados.")

# Deep Graph Library
try:
    import dgl
    from dgl.nn import SAGEConv
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("⚠️ DGL no disponible. GraphSAGE deshabilitado.")

# HuggingFace Transformers
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers no disponible. BERT deshabilitado.")

# XGBoost para Ensemble
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Ensemble usará LightGBM o sklearn.")

# Sklearn fallback
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================

@dataclass
class DeepFraudConfig:
    """Configuración centralizada para todos los módulos."""
    
    # Siamese Network
    siamese_char_embedding_dim: int = 128
    siamese_lstm_hidden: int = 256
    siamese_lstm_layers: int = 2
    siamese_threshold: float = 0.5
    
    # GraphSAGE
    graphsage_hidden_dims: List[int] = None
    graphsage_num_hops: int = 2
    graphsage_aggregator: str = 'mean'
    graphsage_dropout: float = 0.3
    
    # VAE
    vae_hidden_dims: List[int] = None
    vae_latent_dim: int = 16
    vae_beta: float = 1.0  # Weight for KL divergence
    
    # BERT
    bert_model_name: str = 'dccuchile/bert-base-spanish-wwm-cased'
    bert_max_length: int = 256
    bert_num_labels: int = 20  # Número de categorías CNAE principales
    
    # Temporal Fusion
    temporal_hidden_dim: int = 128
    temporal_num_layers: int = 2
    temporal_attention_heads: int = 4
    temporal_sequence_length: int = 5  # 5 años
    
    # Ensemble
    ensemble_n_estimators: int = 200
    ensemble_max_depth: int = 6
    ensemble_learning_rate: float = 0.1
    
    # Device
    device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.graphsage_hidden_dims is None:
            self.graphsage_hidden_dims = [128, 64]
        if self.vae_hidden_dims is None:
            self.vae_hidden_dims = [64, 32]


# ==============================================================================
# MÓDULO 1: SIAMESE ENTITY RESOLVER
# ==============================================================================

if TORCH_AVAILABLE:
    
    class CharacterEmbedding(nn.Module):
        """
        Embeddings a nivel de carácter para nombres de empresas.
        
        RAZÓN FINANCIERA: Los nombres en el M347 vienen con errores tipográficos,
        abreviaturas y variaciones. El char-level captura similitudes que
        el word-level perdería (ej: "S.L." vs "SL" vs "S. L.").
        """
        
        def __init__(self, vocab_size: int = 256, embedding_dim: int = 128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.embedding(x)
    
    
    class SiameseEntityResolver(nn.Module):
        """
        Red Neuronal Siamesa para resolución de entidades.
        
        Detecta que "Transp. Garcia SL" y "Transportes Garcia S.L." son la misma
        entidad, permitiendo unificar nodos en el grafo de transacciones.
        
        Arquitectura:
        - Character Embedding (128 dim)
        - BiLSTM (256 hidden, 2 capas)
        - Contrastive Loss
        
        VALOR FORENSE: Sin esto, el grafo M347 está fragmentado y perdemos
        patrones de carrusel que dependen de identificar correctamente
        las conexiones entre empresas.
        """
        
        def __init__(self, config: DeepFraudConfig = None):
            super().__init__()
            self.config = config or DeepFraudConfig()
            
            # Char-level embedding
            self.char_embedding = CharacterEmbedding(
                vocab_size=256,  # ASCII extendido
                embedding_dim=self.config.siamese_char_embedding_dim
            )
            
            # BiLSTM para secuencia de caracteres
            self.lstm = nn.LSTM(
                input_size=self.config.siamese_char_embedding_dim,
                hidden_size=self.config.siamese_lstm_hidden,
                num_layers=self.config.siamese_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if self.config.siamese_lstm_layers > 1 else 0
            )
            
            # Capa de proyección final
            self.projection = nn.Sequential(
                nn.Linear(self.config.siamese_lstm_hidden * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        
        def encode_string(self, x: torch.Tensor) -> torch.Tensor:
            """Genera embedding para un string."""
            # x: [batch, seq_len] de char indices
            embedded = self.char_embedding(x)  # [batch, seq_len, embed_dim]
            
            # LSTM encoding
            lstm_out, (h_n, _) = self.lstm(embedded)
            
            # Concatenar hidden states de ambas direcciones
            # h_n: [num_layers * 2, batch, hidden]
            h_forward = h_n[-2]  # Última capa, dirección forward
            h_backward = h_n[-1]  # Última capa, dirección backward
            hidden = torch.cat([h_forward, h_backward], dim=1)
            
            # Proyección final
            return self.projection(hidden)
        
        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            """
            Computa distancia entre dos strings.
            
            Args:
                x1: Primer string como tensor de char indices
                x2: Segundo string como tensor de char indices
            
            Returns:
                Distancia euclidiana entre embeddings
            """
            emb1 = self.encode_string(x1)
            emb2 = self.encode_string(x2)
            
            # Distancia euclidiana
            distance = F.pairwise_distance(emb1, emb2)
            return distance
        
        def contrastive_loss(
            self, 
            distance: torch.Tensor, 
            label: torch.Tensor, 
            margin: float = 1.0
        ) -> torch.Tensor:
            """
            Contrastive Loss para entrenamiento.
            
            L = (1-Y) * D² + Y * max(0, margin - D)²
            
            Donde Y=1 si son diferentes, Y=0 si son iguales.
            """
            loss_same = (1 - label) * torch.pow(distance, 2)
            loss_diff = label * torch.pow(torch.clamp(margin - distance, min=0), 2)
            return torch.mean(loss_same + loss_diff)
        
        def should_merge(self, name1: str, name2: str) -> Tuple[bool, float]:
            """
            Determina si dos nombres representan la misma entidad.
            
            Returns:
                (should_merge, distance_score)
            """
            self.eval()
            with torch.no_grad():
                # Convertir strings a tensores
                x1 = self._string_to_tensor(name1)
                x2 = self._string_to_tensor(name2)
                
                distance = self.forward(x1, x2).item()
                should_merge = distance < self.config.siamese_threshold
                
                return should_merge, distance
        
        def _string_to_tensor(self, s: str, max_len: int = 100) -> torch.Tensor:
            """Convierte string a tensor de índices de caracteres."""
            # Normalizar: lowercase, quitar espacios extra
            s = ' '.join(s.lower().split())[:max_len]
            
            # Convertir a índices ASCII
            indices = [ord(c) if ord(c) < 256 else 0 for c in s]
            
            # Padding
            if len(indices) < max_len:
                indices += [0] * (max_len - len(indices))
            
            return torch.tensor([indices], dtype=torch.long, device=self.config.device)


# ==============================================================================
# MÓDULO 2: GRAPHSAGE ENCODER
# ==============================================================================

if TORCH_AVAILABLE and DGL_AVAILABLE:
    
    class GraphSAGEEncoder(nn.Module):
        """
        Encoder basado en GraphSAGE para grafos de transacciones.
        
        Genera embeddings de nodos basados en su vecindario (2 saltos),
        permitiendo detectar "culpabilidad por asociación".
        
        VALOR FORENSE: Una empresa aparentemente sana que tiene conexiones
        estructurales con empresas fraudulentas (mismo proveedor fantasma,
        misma dirección fiscal, etc.) heredará un embedding "sospechoso".
        
        Es INDUCTIVO: puede generar embeddings para empresas nuevas
        que no estaban en el entrenamiento (crucial para producción).
        """
        
        def __init__(
            self, 
            in_feats: int,
            config: DeepFraudConfig = None
        ):
            super().__init__()
            self.config = config or DeepFraudConfig()
            
            hidden_dims = self.config.graphsage_hidden_dims
            
            # Capas SAGEConv (una por cada hop)
            self.layers = nn.ModuleList()
            
            # Primera capa: input -> hidden[0]
            self.layers.append(SAGEConv(
                in_feats, 
                hidden_dims[0], 
                aggregator_type=self.config.graphsage_aggregator
            ))
            
            # Capas intermedias
            for i in range(len(hidden_dims) - 1):
                self.layers.append(SAGEConv(
                    hidden_dims[i], 
                    hidden_dims[i + 1],
                    aggregator_type=self.config.graphsage_aggregator
                ))
            
            self.dropout = nn.Dropout(self.config.graphsage_dropout)
            self.output_dim = hidden_dims[-1]
        
        def forward(
            self, 
            graph: dgl.DGLGraph, 
            features: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass sobre el grafo completo.
            
            Args:
                graph: DGLGraph con estructura de transacciones
                features: Node features [num_nodes, in_feats]
            
            Returns:
                Node embeddings [num_nodes, output_dim]
            """
            h = features
            
            for i, layer in enumerate(self.layers):
                h = layer(graph, h)
                if i < len(self.layers) - 1:  # No activation en última capa
                    h = F.relu(h)
                    h = self.dropout(h)
            
            return h
        
        def get_embeddings_for_new_nodes(
            self,
            graph: dgl.DGLGraph,
            features: torch.Tensor,
            new_node_ids: torch.Tensor
        ) -> torch.Tensor:
            """
            Genera embeddings para nodos nuevos (inferencia inductiva).
            
            Esto es crucial para producción: cuando llega una empresa nueva,
            podemos generar su embedding basándonos en sus conexiones
            con empresas existentes.
            """
            # Forward pass completo
            all_embeddings = self.forward(graph, features)
            
            # Extraer solo los nuevos nodos
            return all_embeddings[new_node_ids]


# ==============================================================================
# MÓDULO 3: FRAUD VAE (Variational Autoencoder)
# ==============================================================================

if TORCH_AVAILABLE:
    
    class FraudVAE(nn.Module):
        """
        Autoencoder Variacional para detección de anomalías.
        
        El VAE aprende la distribución latente de empresas "normales"
        y detecta fraudes como errores de reconstrucción altos.
        
        VALOR FORENSE: Las empresas con maquillaje contable pueden
        manipular cada ratio individualmente para parecer normales,
        pero la COMBINACIÓN de ratios es estadísticamente improbable.
        El VAE captura estas correlaciones no lineales.
        
        Entrenamiento: SOLO con empresas etiquetadas como "No Fraude"
        (Semi-supervisado).
        """
        
        def __init__(
            self, 
            input_dim: int,
            config: DeepFraudConfig = None
        ):
            super().__init__()
            self.config = config or DeepFraudConfig()
            self.input_dim = input_dim
            
            hidden_dims = self.config.vae_hidden_dims
            latent_dim = self.config.vae_latent_dim
            
            # ===== ENCODER =====
            # Comprime features a espacio latente
            encoder_layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = h_dim
            
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Capas para μ y log(σ²)
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
            
            # ===== DECODER =====
            # Reconstruye features desde espacio latente
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = h_dim
            
            decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
            # Sin sigmoid al final para datos normalizados
            
            self.decoder = nn.Sequential(*decoder_layers)
        
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Encode input to latent distribution parameters.
            
            Returns:
                mu: Mean of latent distribution
                logvar: Log variance of latent distribution
            """
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)
        
        def reparameterize(
            self, 
            mu: torch.Tensor, 
            logvar: torch.Tensor
        ) -> torch.Tensor:
            """
            Reparameterization trick para backpropagation.
            
            z = μ + σ * ε, donde ε ~ N(0,1)
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode latent vector to reconstruction."""
            return self.decoder(z)
        
        def forward(
            self, 
            x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Full forward pass.
            
            Returns:
                reconstructed: Reconstrucción de input
                mu: Mean del espacio latente
                logvar: Log-variance del espacio latente
            """
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z)
            return reconstructed, mu, logvar
        
        def loss_function(
            self,
            x: torch.Tensor,
            reconstructed: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            VAE Loss = Reconstruction Loss + β * KL Divergence
            
            KL Divergence regulariza el espacio latente hacia N(0,1).
            β controla el trade-off (β-VAE para mejor disentanglement).
            """
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
            
            # KL Divergence: KL(q(z|x) || p(z))
            # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_loss = recon_loss + self.config.vae_beta * kl_loss
            
            return total_loss, recon_loss, kl_loss
        
        def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
            """
            Calcula score de anomalía basado en error de reconstrucción.
            
            Una empresa fraudulenta tendrá un error alto porque
            su combinación de ratios es improbable en la distribución
            aprendida de empresas sanas.
            """
            self.eval()
            with torch.no_grad():
                reconstructed, _, _ = self.forward(x)
                # Error de reconstrucción por muestra
                recon_error = torch.mean((x - reconstructed) ** 2, dim=1)
                return recon_error


# ==============================================================================
# MÓDULO 4: BERT SEMANTIC ANALYZER
# ==============================================================================

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    
    class BERTSemanticAnalyzer(nn.Module):
        """
        Analizador semántico basado en BERT español.
        
        Detecta inconsistencias entre el texto del "Objeto Social" 
        y la actividad económica declarada (CNAE).
        
        VALOR FORENSE: Una empresa que dice dedicarse a "hostelería"
        en su memoria pero declara CNAE de "construcción" y tiene
        transacciones M347 con proveedores de materiales indica
        incoherencia que sugiere fraude fiscal.
        """
        
        def __init__(self, config: DeepFraudConfig = None):
            super().__init__()
            self.config = config or DeepFraudConfig()
            
            # Cargar tokenizer y modelo BERT español
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.bert_model_name
            )
            
            self.bert = AutoModel.from_pretrained(
                self.config.bert_model_name
            )
            
            # Clasificador sobre embeddings CLS
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),  # BERT base hidden size = 768
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.config.bert_num_labels)
            )
        
        def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass para clasificación de CNAE desde texto.
            
            Args:
                input_ids: Token IDs del texto [batch, seq_len]
                attention_mask: Máscara de atención [batch, seq_len]
            
            Returns:
                Logits de clasificación [batch, num_labels]
            """
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Usar embedding del token [CLS]
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            return self.classifier(cls_embedding)
        
        def get_semantic_inconsistency_score(
            self,
            text: str,
            declared_cnae: int
        ) -> Tuple[float, int]:
            """
            Calcula score de inconsistencia semántica.
            
            Args:
                text: Objeto social o descripción de actividad
                declared_cnae: CNAE declarado por la empresa
            
            Returns:
                (inconsistency_score, predicted_cnae)
            """
            self.eval()
            with torch.no_grad():
                # Tokenizar texto
                encoded = self.tokenizer(
                    text,
                    max_length=self.config.bert_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.config.device)
                attention_mask = encoded['attention_mask'].to(self.config.device)
                
                # Predicción
                logits = self.forward(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                
                predicted_cnae = torch.argmax(probs, dim=1).item()
                
                # Score de inconsistencia: 1 - P(declared_cnae | text)
                if declared_cnae < self.config.bert_num_labels:
                    consistency_prob = probs[0, declared_cnae].item()
                else:
                    consistency_prob = 0.5  # CNAE desconocido
                
                inconsistency_score = 1.0 - consistency_prob
                
                return inconsistency_score, predicted_cnae


# ==============================================================================
# MÓDULO 5: TEMPORAL FUSION MODULE
# ==============================================================================

if TORCH_AVAILABLE:
    
    class TemporalAttention(nn.Module):
        """
        Mecanismo de atención sobre secuencias temporales.
        
        Identifica qué períodos del pasado tienen mayor peso
        en el riesgo actual de fraude.
        """
        
        def __init__(self, hidden_dim: int, num_heads: int = 4):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
        
        def forward(
            self, 
            x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: [batch, seq_len, hidden_dim]
            
            Returns:
                attended: [batch, seq_len, hidden_dim]
                attention_weights: [batch, seq_len, seq_len]
            """
            attended, weights = self.attention(x, x, x)
            attended = self.norm(attended + x)  # Residual connection
            return attended, weights
    
    
    class TemporalFusionModule(nn.Module):
        """
        Módulo de Fusión Temporal con LSTM y Atención.
        
        Detecta cambios abruptos en patrones temporales, típicos
        del fraude "bust-out" o vaciado de empresa.
        
        VALOR FORENSE: El fraude "bust-out" sigue un patrón característico:
        1. Años de comportamiento normal (construir confianza)
        2. Cambio de administrador
        3. Incremento repentino de deuda
        4. Vaciado de activos
        
        El mecanismo de atención identifica CUÁNDO empezó el fraude.
        """
        
        def __init__(
            self, 
            input_dim: int,  # Features por timestep (Ventas, Deuda, Caja, etc.)
            config: DeepFraudConfig = None
        ):
            super().__init__()
            self.config = config or DeepFraudConfig()
            
            # LSTM bidireccional para capturar contexto temporal
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.config.temporal_hidden_dim,
                num_layers=self.config.temporal_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if self.config.temporal_num_layers > 1 else 0
            )
            
            # Atención temporal
            self.attention = TemporalAttention(
                hidden_dim=self.config.temporal_hidden_dim * 2,  # Bidirectional
                num_heads=self.config.temporal_attention_heads
            )
            
            # Clasificador final
            self.classifier = nn.Sequential(
                nn.Linear(self.config.temporal_hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(
            self, 
            x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Secuencia temporal [batch, seq_len, input_dim]
                   Ej: 5 años de [Ventas, Deuda, Caja, Margen]
            
            Returns:
                risk_score: Probabilidad de fraude [batch, 1]
                attention_weights: Pesos de atención [batch, seq_len, seq_len]
            """
            # LSTM encoding
            lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
            
            # Atención temporal
            attended, attention_weights = self.attention(lstm_out)
            
            # Usar último timestep para clasificación
            last_hidden = attended[:, -1, :]
            
            # Score de riesgo
            risk_score = self.classifier(last_hidden)
            
            return risk_score, attention_weights
        
        def get_temporal_risk(
            self,
            time_series: torch.Tensor
        ) -> Tuple[float, np.ndarray]:
            """
            Calcula riesgo temporal y peso de cada período.
            
            Returns:
                (risk_score, attention_weights_per_year)
            """
            self.eval()
            with torch.no_grad():
                if time_series.dim() == 2:
                    time_series = time_series.unsqueeze(0)
                
                risk, attention = self.forward(time_series)
                
                # Promediar attention weights para interpretabilidad
                avg_attention = attention.mean(dim=-1).squeeze().cpu().numpy()
                
                return risk.item(), avg_attention


# ==============================================================================
# ORQUESTADOR: DEEP FRAUD ENSEMBLE
# ==============================================================================

class DeepFraudEnsemble:
    """
    Orquestador que combina outputs de todos los módulos Deep Learning
    en un clasificador XGBoost final.
    
    El "Super-Vector" incluye:
    - GraphSAGE embeddings (64 dim)
    - VAE reconstruction error (1 dim)
    - BERT semantic score (1 dim)
    - Temporal risk score (1 dim)
    - Features originales (~20 dim)
    
    Total: ~87 features para el clasificador final.
    """
    
    def __init__(self, config: DeepFraudConfig = None):
        self.config = config or DeepFraudConfig()
        self.scaler = StandardScaler()
        
        # Inicializar modelos (None si dependencias no disponibles)
        self.entity_resolver = None
        self.graph_encoder = None
        self.vae = None
        self.bert_analyzer = None
        self.temporal_module = None
        
        # Clasificador final
        if XGB_AVAILABLE:
            self.final_classifier = xgb.XGBClassifier(
                n_estimators=self.config.ensemble_n_estimators,
                max_depth=self.config.ensemble_max_depth,
                learning_rate=self.config.ensemble_learning_rate,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        else:
            self.final_classifier = GradientBoostingClassifier(
                n_estimators=self.config.ensemble_n_estimators,
                max_depth=self.config.ensemble_max_depth,
                learning_rate=self.config.ensemble_learning_rate,
                random_state=42
            )
        
        self.is_fitted = False
    
    def initialize_modules(
        self,
        input_dim_vae: int = 20,
        input_dim_graphsage: int = 10,
        input_dim_temporal: int = 4
    ):
        """Inicializa todos los módulos con las dimensiones correctas."""
        
        if TORCH_AVAILABLE:
            self.entity_resolver = SiameseEntityResolver(self.config)
            self.vae = FraudVAE(input_dim_vae, self.config)
            self.temporal_module = TemporalFusionModule(input_dim_temporal, self.config)
            
            if DGL_AVAILABLE:
                self.graph_encoder = GraphSAGEEncoder(input_dim_graphsage, self.config)
            
            if TRANSFORMERS_AVAILABLE:
                self.bert_analyzer = BERTSemanticAnalyzer(self.config)
        
        return self
    
    def build_super_vector(
        self,
        features: np.ndarray,
        graph_embeddings: Optional[np.ndarray] = None,
        vae_scores: Optional[np.ndarray] = None,
        bert_scores: Optional[np.ndarray] = None,
        temporal_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Construye el super-vector concatenando todos los outputs.
        
        Args:
            features: Features originales [n_samples, n_features]
            graph_embeddings: GraphSAGE output [n_samples, 64]
            vae_scores: VAE anomaly scores [n_samples, 1]
            bert_scores: BERT inconsistency [n_samples, 1]
            temporal_scores: Temporal risk [n_samples, 1]
        
        Returns:
            Super-vector [n_samples, total_dim]
        """
        components = [features]
        
        if graph_embeddings is not None:
            components.append(graph_embeddings)
        
        if vae_scores is not None:
            if vae_scores.ndim == 1:
                vae_scores = vae_scores.reshape(-1, 1)
            components.append(vae_scores)
        
        if bert_scores is not None:
            if bert_scores.ndim == 1:
                bert_scores = bert_scores.reshape(-1, 1)
            components.append(bert_scores)
        
        if temporal_scores is not None:
            if temporal_scores.ndim == 1:
                temporal_scores = temporal_scores.reshape(-1, 1)
            components.append(temporal_scores)
        
        return np.hstack(components)
    
    def fit(
        self,
        super_vector: np.ndarray,
        labels: np.ndarray
    ) -> 'DeepFraudEnsemble':
        """Entrena el clasificador final sobre el super-vector."""
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(super_vector)
        
        # Entrenar clasificador
        self.final_classifier.fit(X_scaled, labels)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, super_vector: np.ndarray) -> np.ndarray:
        """Predice probabilidad de fraude."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble no entrenado. Llama fit() primero.")
        
        X_scaled = self.scaler.transform(super_vector)
        return self.final_classifier.predict_proba(X_scaled)[:, 1]
    
    def predict(
        self, 
        super_vector: np.ndarray, 
        threshold: float = 0.5
    ) -> np.ndarray:
        """Predice etiquetas binarias."""
        probs = self.predict_proba(super_vector)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importancia de features del clasificador final."""
        if not self.is_fitted:
            return {}
        
        if XGB_AVAILABLE:
            importance = self.final_classifier.feature_importances_
        else:
            importance = self.final_classifier.feature_importances_
        
        # Crear nombres genéricos
        names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(names, importance))
    
    def get_model_status(self) -> Dict[str, bool]:
        """Retorna estado de disponibilidad de cada módulo."""
        return {
            'entity_resolver': self.entity_resolver is not None,
            'graph_encoder': self.graph_encoder is not None,
            'vae': self.vae is not None,
            'bert_analyzer': self.bert_analyzer is not None,
            'temporal_module': self.temporal_module is not None,
            'final_classifier': self.is_fitted,
            'torch_available': TORCH_AVAILABLE,
            'dgl_available': DGL_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'xgboost_available': XGB_AVAILABLE
        }


# ==============================================================================
# UTILIDADES SPARK (Para inferencia distribuida)
# ==============================================================================

def create_vae_inference_udf(vae_model: 'FraudVAE'):
    """
    Crea un pandas_udf para inferencia distribuida del VAE en Spark.
    
    Uso:
        from pyspark.sql.functions import pandas_udf
        
        vae_udf = create_vae_inference_udf(trained_vae)
        df = df.withColumn('anomaly_score', vae_udf(*feature_columns))
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch requerido para inferencia VAE")
    
    # Serializar modelo para broadcast
    import io
    buffer = io.BytesIO()
    torch.save(vae_model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    
    def predict_anomaly(iterator):
        """UDF que procesa batches de pandas."""
        # Cargar modelo una vez por worker
        model = FraudVAE(vae_model.input_dim, vae_model.config)
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer))
        model.eval()
        
        for pdf in iterator:
            X = torch.tensor(pdf.values, dtype=torch.float32)
            with torch.no_grad():
                scores = model.get_anomaly_score(X).numpy()
            yield pd.DataFrame({'anomaly_score': scores})
    
    return predict_anomaly


# ==============================================================================
# DEMO Y TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DEEP FRAUD HUNTER - Framework de Deep Learning")
    print("=" * 80)
    
    # Verificar disponibilidad de módulos
    config = DeepFraudConfig()
    ensemble = DeepFraudEnsemble(config)
    
    print("\n[1] Estado de Módulos:")
    status = ensemble.get_model_status()
    for module, available in status.items():
        icon = "✅" if available else "❌"
        print(f"    {icon} {module}: {available}")
    
    # Demo con datos sintéticos
    print("\n[2] Demo con datos sintéticos...")
    
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Generar features
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Simular outputs de módulos
    graph_embeddings = np.random.randn(n_samples, 64)
    vae_scores = np.random.rand(n_samples)
    bert_scores = np.random.rand(n_samples)
    temporal_scores = np.random.rand(n_samples)
    
    # Construir super-vector
    super_vector = ensemble.build_super_vector(
        features=X,
        graph_embeddings=graph_embeddings,
        vae_scores=vae_scores,
        bert_scores=bert_scores,
        temporal_scores=temporal_scores
    )
    
    print(f"    Super-vector shape: {super_vector.shape}")
    
    # Entrenar ensemble
    ensemble.fit(super_vector, y)
    print("    ✅ Ensemble entrenado")
    
    # Predicciones
    probs = ensemble.predict_proba(super_vector)
    print(f"    Probabilidades: min={probs.min():.3f}, max={probs.max():.3f}")
    
    # Test Siamese (si disponible)
    if TORCH_AVAILABLE:
        print("\n[3] Test SiameseEntityResolver:")
        resolver = SiameseEntityResolver(config)
        
        test_pairs = [
            ("Transportes Garcia SL", "Transp. Garcia S.L."),
            ("Construcciones ABC", "Hosteleria XYZ"),
            ("INVERSIONES LOPEZ SA", "Inversiones Lopez S.A."),
        ]
        
        for name1, name2 in test_pairs:
            should_merge, distance = resolver.should_merge(name1, name2)
            icon = "🔗" if should_merge else "✂️"
            print(f"    {icon} '{name1}' vs '{name2}'")
            print(f"       Distance: {distance:.4f}, Merge: {should_merge}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETADA")
    print("=" * 80)
