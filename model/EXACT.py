"""EXACT model pipeline shared by the daily and minute forecasting protocols."""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) not divisible by heads ({n_heads})"
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.hidden_act = hidden_act

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        if self.hidden_act == "gelu":
            hidden_states = F.gelu(hidden_states)
        elif self.hidden_act == "relu":
            hidden_states = F.relu(hidden_states)
        else:
            hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        hidden_act="gelu",
    ):
        super().__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class WhiteBoxBackbone(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(
            n_layers=num_layers,
            n_heads=nhead,
            hidden_size=d_model,
            inner_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout,
            hidden_act="gelu",
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, attention_mask=None)
        return x[:, -1, :]


class ContextGatingRouter(nn.Module):
    def __init__(self, d_model=64, state_dim=2, time_dim=4, topo_dim=16, dropout=0.1):
        super().__init__()
        input_dim = state_dim + time_dim + topo_dim + (2 * d_model)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, state_features, time_features, topo_emb, h_fine, h_coarse):
        context = torch.cat(
            [state_features, time_features, topo_emb, h_fine, h_coarse], dim=-1
        )
        gate = self.gate_net(context)
        return gate


class EXACT(nn.Module):
    """Forecast log realized variance with topology-conditioned fine/coarse fusion.

    Inputs are raw and persistent windows (batch, sequence, channels),
    topology descriptors (batch, 2 * topo_bins), state features (batch, 2),
    and cyclical time features (batch, 4). Returns the forecast, routing
    gate, and reconstruction/persistent/high-frequency/noise/topology outputs.
    """

    def __init__(
        self,
        channels=2,
        d_model=64,
        *,
        seq_len=30,
        nhead=2,
        num_layers=2,
        dropout=0.1,
        topo_bins=15,
        topo_emb_dim=16,
        router_dropout=0.1,
    ):
        super().__init__()
        self.topo_proj = nn.Sequential(
            nn.Linear(topo_bins * 2, topo_emb_dim),
            nn.GELU(),
            nn.LayerNorm(topo_emb_dim),
        )
        self.fine_backbone = WhiteBoxBackbone(
            input_dim=channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.downsample = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        self.coarse_backbone = WhiteBoxBackbone(
            input_dim=channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.router = ContextGatingRouter(
            d_model=d_model,
            state_dim=2,
            time_dim=4,
            topo_dim=topo_emb_dim,
            dropout=router_dropout,
        )
        self.topo_scale = nn.Linear(topo_emb_dim, d_model)
        self.topo_shift = nn.Linear(topo_emb_dim, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_dropout = nn.Dropout(dropout)
        self.pred_head = nn.Linear(d_model, 1)
        self.residual_weight = nn.Parameter(torch.tensor(2.0))
        self.reco_head = nn.Linear(d_model, seq_len)
        self.persistent_head = nn.Linear(d_model, seq_len)
        self.highfreq_head = nn.Linear(d_model, seq_len)
        self.noise_head = nn.Linear(d_model, seq_len)
        self.topo_pred_head = nn.Linear(d_model, topo_bins * 2)

    def forward(self, raw_x, persistent_input, topo_feat, state_feat, time_feat):
        topo_emb = self.topo_proj(topo_feat)
        h_fine = self.fine_backbone(raw_x)
        x_coarse = self.downsample(persistent_input.transpose(1, 2)).transpose(1, 2)
        h_coarse = self.coarse_backbone(x_coarse)

        gate = self.router(state_feat, time_feat, topo_emb, h_fine, h_coarse)
        fused_expert = gate * h_fine + (1 - gate) * h_coarse

        scale = torch.tanh(self.topo_scale(topo_emb))
        shift = self.topo_shift(topo_emb)
        modulated = fused_expert * (1 + scale) + shift

        fused = self.fusion_dropout(self.fusion_norm(modulated))

        pred = (
            self.pred_head(fused)
            + torch.sigmoid(self.residual_weight) * raw_x[:, -1, 0:1]
        )
        reco = self.reco_head(fused)
        persistent = self.persistent_head(fused)
        highfreq = self.highfreq_head(fused)
        noise = self.noise_head(fused)
        topo_pred = self.topo_pred_head(fused)
        return pred, gate, reco, persistent, highfreq, noise, topo_pred
