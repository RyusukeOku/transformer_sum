import torch
import torch.nn as nn
import math
from typing import Optional # 追加

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 input_vocab_size: int,
                 output_vocab_size: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout_p: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(input_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_p, max_seq_len)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout_p,
                                          batch_first=False)

        self.generator = nn.Linear(d_model, output_vocab_size)

    def _apply_embedding_and_pe(self, tokens: torch.Tensor, embedding_layer: nn.Embedding) -> torch.Tensor:
        embedded_tokens = embedding_layer(tokens) * math.sqrt(self.d_model)
        return self.positional_encoding(embedded_tokens)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor], src_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        src_emb = self._apply_embedding_and_pe(src, self.src_tok_emb)
        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor],
               tgt_padding_mask: Optional[torch.Tensor], memory_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        tgt_emb = self._apply_embedding_and_pe(tgt, self.tgt_tok_emb)
        output = self.transformer.decoder(tgt_emb, memory,
                                          tgt_mask=tgt_mask,
                                          memory_mask=None,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        memory = self.encode(src, src_mask, src_padding_mask)
        decoder_output = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        logits = self.generator(decoder_output)
        return logits

# (transformer.py の残りの部分は元のままとします。main関数などがあればそれも同様です)
# 例として、以前の main 関数部分は削除されているか、このクラス定義とは独立していると仮定します。
# もし transformer.py にテスト用の main 関数があれば、それはこのクラス変更に合わせて更新が必要かもしれません。