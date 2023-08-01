import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import get_activation
from transformers.modeling_outputs import BaseModelOutput

INF = 1e4


def attention_normalize(a, mask=None, dim=-1, method="softmax"):

    if method == "softmax":
        return torch.softmax(a, dim=dim)
    else:
        if mask is not None:
            assert mask.ndim == 3
            l = mask.sum(-1, keepdim=True)
        else:
            l = torch.ones_like(a) * a.shape[-2]
        if method == "squared_relu":
            return torch.relu(a) ** 2 / l
        elif method == "softmax_plus":
            scale = torch.log(l) / np.log(512)
            # mask: 1 for not padding, 0 for padding
            # padding position's scale is 1
            if mask is not None:
                scale = scale.masked_fill(mask == 0, 1.0)
            return torch.softmax(a * scale, dim=dim)
    return a


class ScaleOffset(nn.Module):


    def __init__(
        self,
        hidden_size=768,
        scale=True,
        offset=True,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset

        if self.scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if self.offset:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs):
        if self.scale:
            inputs = inputs * self.weight
        if self.offset:
            inputs = inputs + self.bias

        return inputs


class Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


class GatedAttentionUnit(nn.Module):


    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        attention_key_size=128,
        activation="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.intermediate_size = intermediate_size
        self.attention_key_size = attention_key_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout

        self.i_dense = nn.Linear(
            hidden_size, 2 * intermediate_size + attention_key_size, bias=self.use_bias
        )
        self.o_dense = nn.Linear(intermediate_size, hidden_size, bias=self.use_bias)

        self.q_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        # x.shape [batch, seq_len, 2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):
        x = self.i_dense(hidden_states)
        u, v, qk = torch.split(
            self.activation(x),
            [self.intermediate_size, self.intermediate_size, self.attention_key_size],
            dim=-1,
        )
        q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)

        q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(
            k, sinusoidal_pos
        )

        # Attention
        a = torch.einsum("bmd,bnd->bmn", q, k)

        if self.attention_scale:
            a = a / self.attention_key_size ** 0.5

        if attention_mask is not None:
            a = a.masked_fill(attention_mask == 0, -INF)

        A = attention_normalize(a, attention_mask, dim=-1, method=self.normalization)

        A = F.dropout(A, p=self.attention_dropout, training=self.training)

        o = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))

        outputs = (o, A) if output_attentions else (o,)
        return outputs


class GAULayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        attention_key_size=128,
        activation="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        eps=1e-12,
    ):
        super().__init__()
        self.gau = GatedAttentionUnit(
            hidden_size,
            intermediate_size,
            attention_key_size,
            activation,
            use_bias,
            normalization,
            attention_scale,
            attention_dropout,
        )
        self.norm = Norm(eps=eps)
        self.hidden_dropout = hidden_dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):
        gau_output = self.gau(
            hidden_states, attention_mask, sinusoidal_pos, output_attentions
        )

        # dropout and residual
        o = F.dropout(gau_output[0], p=self.hidden_dropout, training=self.training)
        o = self.norm(hidden_states + o)

        outputs = (o,) + gau_output[1:]  # add attentions if we output them

        return outputs

class GAUEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                GAULayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    attention_key_size=config.attention_key_size,
                    activation=config.hidden_act,
                    use_bias=config.use_bias,
                    normalization=config.normalization,
                    attention_scale=config.attention_scale,
                    attention_dropout=config.attention_probs_dropout_prob,
                    hidden_dropout=config.hidden_dropout_prob,
                    eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        sinusoidal_id = self.get_sinusoidal_id(
            config.max_position_embeddings, config.attention_key_size
        )
        self.register_buffer("sin_pos", sinusoidal_id.sin(), persistent=False)
        self.register_buffer("cos_pos", sinusoidal_id.cos(), persistent=False)
    
    def get_sinusoidal_id(self, max_length, output_dim):
        position_ids = torch.arange(0, max_length, dtype=torch.float32)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float32)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        sinusoidal_id = torch.einsum("n,d->nd", position_ids, indices)
        return sinusoidal_id[None, :, :]
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        seqlen = hidden_states.shape[1]
        sinusoidal_pos = self.sin_pos[:, :seqlen, :], self.cos_pos[:, :seqlen, :]
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GAUConfig(PretrainedConfig):
    model_type = "gau"
    
    def __init__(
            self,
            vocab_size=12000,
            hidden_size=768,
            intermediate_size=1536,
            num_hidden_layers=24,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            attention_key_size=128,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            hidden_act="swish",
            classifier_dropout=0.1,
            use_bias=False,
            normalization="softmax",
            attention_scale=True,
            embedding_size=None,
            scaling_factor="n",
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_key_size = attention_key_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.gradient_checkpointing = gradient_checkpointing
        self.classifier_dropout = classifier_dropout
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.intermediate_size = intermediate_size
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
        self.scaling_factor = scaling_factor
