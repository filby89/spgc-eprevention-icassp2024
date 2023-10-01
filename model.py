import math
import torch
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):

    """
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).
    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """

    def __init__(self, args): 

        """
        Args:
            input_features: int, number of input variables. 1 if univariate.
            seq_len: int, the length of the input sequence fed to the model
            d_model: int, All sub-layers in the model produce 
                     outputs of dimension d_model
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_head: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
        """

        super().__init__() 

        args_defaults = dict(
            input_features = 10,
            seq_len = 512,
            batch_first = False,
            device = 'cuda',
            d_model = 32,  
            n_encoder_layers = 2,
            n_decoder_layers = 2,
            n_head = 8,
            dropout_encoder = 0.2, 
            dropout_pos_enc = 0.1,
            dim_feedforward_encoder = 2048,
            num_patients = 10
        )

        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        # Input Embedding (Encoder)
        self.encoder_input_layer = nn.Linear(in_features=self.input_features, out_features=self.d_model)
        
        # Output Mapping (Linear Reconstruction Layer)
        self.linear_mapping = nn.Linear(in_features=self.d_model, out_features=self.input_features)

        # Positional Encoding
        self.positional_encoding_layer = PositionalEncoding(d_model=self.d_model, dropout=self.dropout_pos_enc)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward_encoder,
            dropout=self.dropout_encoder,
            batch_first=self.batch_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_encoder_layers, 
        )
        
        # classifier
        self.adaptive = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.d_model, self.num_patients)

    def forward(self, src: Tensor) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
        """
        src = src.permute(2, 0, 1)

        # Input Embedding (Encoder)
        src = self.encoder_input_layer(src) 

        # Positional Encoding (Encoder)
        src = self.positional_encoding_layer(src) 

        # Encoder
        src = self.encoder(src=src)

        features = src.permute(1, 2, 0).contiguous()
        features = self.adaptive(features).squeeze(-1)
        logits = self.classifier(features)    

        return logits, features