import copy
from abc import ABC
from dataclasses import dataclass
from typing import Union, Tuple, Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RoFormerPreTrainedModel, RoFormerModel, BertModel, BertPreTrainedModel, PreTrainedModel, \
    RoFormerConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding, RoFormerEncoder
from transformers.utils import ModelOutput
from transformers import BertModel

from gau_layer import GAUEncoder, GAUConfig
from model_flash_v4 import EhrModel


class RoFormerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, project_dim=512, scale=1):
        super().__init__()
        hidden_size = project_dim
        self.out_proj = nn.Linear(hidden_size * scale, config.num_labels)

        self.config = config

    def forward(self, x, **kwargs):
        x = self.out_proj(x)
        return x


class RoFormerForSequenceCls(PreTrainedModel):
    config_class = RoFormerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_load_unexpected = [
        "roformer.encoder.embed_positions.weight"
    ]

    def __init__(self, config, projection_dim=768, eos_token_id=102, scale=2):
        super(RoFormerForSequenceCls, self).__init__(config)
        self.model = EhrModel(config, projection_dim, eos_token_id)
        self.fc = RoFormerClassificationHead(config, project_dim=projection_dim, scale=3)
        # self.fusion_att = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, batch_first=True, norm_first=True, activation=F.gelu),
        #     num_layers=6
        # )
        self.post_init()
        self.model.logit_scale.requires_grad = False
        self.model.exam_mlm_head = None
        # self.model.requires_grad_(False)
        self.model.fusion_projection = None
        self.model.diag_projection = None
        self.model.drug_projection = None

    def set_exam_labels(self, input_ids, attention_mask):
        exam_input_hiddens = self.model.get_text_hiddens(
            input_ids, attention_mask,
            return_dict=False
        )
        self.model.exam_input_hiddens = nn.Parameter(exam_input_hiddens[:, 0].unsqueeze(0), requires_grad=False)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RoFormerEncoder):
            module.gradient_checkpointing = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            exam_input_values=None, exam_value_mask=None,
            diag_input_ids=None,
            diag_attention_mask=None,
            drug_input_ids=None,
            drug_attention_mask=None,
            labels=None,
            return_loss=True,
            return_dict=None,
    ):
        batch_size = input_ids.shape[0]
        # exam_value_mask[:, 1:] = 1

        exam_outputs = self.model.get_exam_outputs(exam_input_values, exam_value_mask)
        fusion_outputs, fusion_features = self.model.get_fusion_features(input_ids, attention_mask, exam_outputs,
                                                                         exam_value_mask)
        diag_outputs, diag_features = self.model.get_diag_features(diag_input_ids, diag_attention_mask)
        drug_outputs, drug_features = self.model.get_drug_features(drug_input_ids, drug_attention_mask)

        # sequence_output = torch.concat((
        #     fusion_features, diag_features, drug_features
        # ), -1)
        sequence_output = torch.concat((
            fusion_outputs[:, 0], diag_outputs[:, 0], drug_outputs[:, 0]
        ), -1)


        logits = self.fc(sequence_output)

        # logits = self.fc(
        #     note_features
        # )

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(weight=torch.tensor([1, 50]).float().to(logits.device))
            loss_fct = CrossEntropyLoss()
            # loss_fct = BCEFocalLosswithLogits()
            loss = loss_fct(logits, labels.long())


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output,
        )

