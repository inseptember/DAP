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


class ExamPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size=768, layer_norm_eps=1e-5):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ExamPredictionHead(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=105):
        super().__init__()
        self.transform = ExamPredictionHeadTransform(hidden_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ExamEncoder(PreTrainedModel):
    def __init__(self, config, layer_num=6, exam_num=150):
        super(ExamEncoder, self).__init__(config)
        # self.emb_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.trans = FLASHTransformer(dim=config.hidden_size, layer_num=layer_num)
        self.embeddings = nn.Embedding(num_embeddings=exam_num, embedding_dim=config.hidden_size, padding_idx=103)
        self.trans = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.num_attention_heads, batch_first=True, norm_first=True,
                layer_norm_eps=config.layer_norm_eps, activation=config.hidden_act
            ),
            num_layers=layer_num
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.trans = GAUEncoder(
        #     num_hidden_layers=layer_num, hidden_size=config.hidden_size, eps=config.layer_norm_eps
        # )
        # self.trans = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, exam_ids, exam_mask, exam_label_hiddens=None):
        # input_shape = exam_vals.size()
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(exam_mask, input_shape)
        exam_hidden = self.embeddings(exam_ids.int())
        batch = exam_hidden.shape[0]
        device = exam_hidden.device
        # exam_ids = torch.cat((torch.ones((batch, 1)).to(device) * 202, exam_vals), dim=1).long()
        # exam_mask = torch.cat((torch.ones((batch, 1)).to(device), exam_mask), dim=1)
        # hidden = exam_hidden + exam_label_hiddens
        # hidden = self.dropout(self.emb_norm(hidden))
        if exam_label_hiddens is not None:
            hidden = exam_hidden + exam_label_hiddens
        else:
            hidden = exam_hidden
        hidden = self.trans(hidden, src_key_padding_mask=(exam_mask == 0))
        hidden = self.norm(hidden)
        # hidden = self.trans(exam_hidden, attention_mask=exam_mask[:, None, :])[0]
        return hidden


@dataclass
class MultiOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_diag: torch.FloatTensor = None
    logits_per_note: torch.FloatTensor = None
    diag_embeds: torch.FloatTensor = None
    note_embeds: torch.FloatTensor = None
    drug_embeds: torch.FloatTensor = None
    diag_model_output: BaseModelOutputWithPooling = None
    drug_model_output: BaseModelOutputWithPooling = None
    note_model_output: BaseModelOutputWithPooling = None


def contrastive_loss(logits: torch.Tensor, mask=None) -> torch.Tensor:
    labels = torch.arange(len(logits)).to(logits.device)
    if mask is not None:
        labels = labels.masked_fill(mask, -100)
    return nn.functional.cross_entropy(logits, labels)
    # return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor, mask=None) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, mask)
    image_loss = contrastive_loss(similarity.t(), mask)
    return (caption_loss + image_loss) / 2.0


class DiagEncoder(RoFormerModel, ABC):
    def __init__(self, config):
        super(DiagEncoder, self).__init__(config)


class DrugEncoder(RoFormerModel, ABC):
    def __init__(self, config):
        super(DrugEncoder, self).__init__(config)


def copy_cfg(cfg: RoFormerConfig, **kwargs):
    config = GAUConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=kwargs.get('intermediate_size', 1536),
        num_hidden_layers=kwargs.get('num_hidden_layers', 24),
        max_position_embeddings=2600,
        type_vocab_size=2,
        initializer_range=0.02,
        attention_key_size=128,
        layer_norm_eps=cfg.layer_norm_eps,
        pad_token_id=cfg.pad_token_id,
        gradient_checkpointing=False,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        hidden_act="swish",
        classifier_dropout=0.1,
        use_bias=False,
        normalization="softmax",
        attention_scale=True,
        embedding_size=None,
        scaling_factor="n",
    )
    return config


class EhrModel(PreTrainedModel):
    config_class = RoFormerConfig
    base_model_prefix = "roformer"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_load_unexpected = [
        r"roformer.embeddings_project.weight",
        r"roformer.embeddings_project.bias",
    ]

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

    def __init__(self, config, projection_dim=768, eos_token_id=102, mask_token_id=103, exam_num=150):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.note_encoder = GAUEncoder(copy_cfg(config))
        self.exam_encoder = ExamEncoder(config, layer_num=6, exam_num=exam_num)
        self.diag_former = GAUEncoder(copy_cfg(config))

        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.exam_num = exam_num
        # self.note_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.fusion_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.diag_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.drug_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)

        logit_scale = torch.ones([3]) * 2.6592
        logit_scale[-1] = 0
        self.logit_scale = nn.Parameter(logit_scale)

        self.mlm_probability = 0.15

        self.exam_mlm_head = ExamPredictionHead(config.hidden_size, 150)


        # self.fusion_att = nn.MultiheadAttention(
        #     embed_dim=config.hidden_size, num_heads=config.num_attention_heads, batch_first=True
        # )
        # self.fusion_att = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, batch_first=True,
        #                                norm_first=True, activation=F.gelu),
        #     num_layers=6
        # )
        # Initialize weights and apply final processing
        self.post_init()

        self.roformer = None

        self.exam_input_hiddens = None

    def set_input_encoder(self):
        self.roformer = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base", max_position_embeddings=2600)
        self.roformer.requires_grad_(False)

    def set_exam_labels(self, input_ids, attention_mask):
        exam_input_hiddens = self.get_text_hiddens(
            input_ids, attention_mask,
            return_dict=False
        )
        self.exam_input_hiddens = nn.Parameter(exam_input_hiddens[:, 0].unsqueeze(0), requires_grad=False)

    def get_text_hiddens(self, input_ids, attention_mask, return_dict=False):
        with torch.no_grad():
            outputs = self.roformer(
                input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            return outputs[0]

    def get_diag_encoder(self):
        return self.diag_former

    def get_drug_encoder(self):
        return self.diag_former

    def get_note_encoder(self):
        return self.note_encoder

    def get_diag_features(self, input_ids, attention_mask):
        outputs = self.get_diag_encoder()(
            self.get_text_hiddens(
                input_ids, attention_mask
            ),
            attention_mask[:, None, :]
        )
        sequence_output = outputs[0]

        features = None
        if self.diag_projection is not None:
            features = self.diag_projection(sequence_output[:, 0])
        return outputs[0], features

    def get_exam_outputs(self, exam_input_values=None, exam_value_mask=None):
        exam_input_hiddens = None
        if self.exam_input_hiddens is not None:
            exam_input_hiddens = self.exam_input_hiddens.repeat((exam_input_values.shape[0], 1, 1))

        exam_outputs = self.exam_encoder(exam_input_values, exam_value_mask, exam_input_hiddens)
        return exam_outputs

    def get_fusion_features(self, input_ids=None, attention_mask=None, exam_outputs=None, exam_value_mask=None):
        encoder = self.get_note_encoder()
        outputs = encoder(
            torch.cat((
                exam_outputs,
                self.get_text_hiddens(
                    input_ids, attention_mask
                )
            ), dim=1),
            torch.cat((
                exam_value_mask,
                attention_mask
            ), dim=1)[:, None, :]
        )
        sequence_output = outputs[0]

        features = None
        if self.fusion_projection is not None:
            features = self.fusion_projection(sequence_output[:, 0])
        return sequence_output, features

    def get_drug_features(self, input_ids, attention_mask):
        outputs = self.get_drug_encoder()(
            self.get_text_hiddens(
                input_ids, attention_mask
            ),
            attention_mask[:, None, :]
        )
        sequence_output = outputs[0]

        features = None
        if self.drug_projection is not None:
            features = self.drug_projection(sequence_output[:, 0])
        return outputs[0], features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            diag_input_ids=None,
            diag_attention_mask=None,
            drug_input_ids=None,
            drug_attention_mask=None,
            exam_input_values=None, exam_value_mask=None,
            return_loss=True,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        exam_input_labels = None
        if self.training:
            exam_input_labels = exam_input_values.clone()
            probability_matrix = torch.full(exam_input_values.shape, self.mlm_probability).to(exam_input_values.device)
            probability_matrix.masked_fill_(exam_value_mask == 0, value=0.0)
            probability_matrix[:, 0] = 0
            masked_indices = torch.bernoulli(probability_matrix).bool()
            indices_replaced = torch.bernoulli(torch.full(exam_input_values.shape, 0.8)).bool().to(exam_input_values.device) & masked_indices
            exam_input_values[indices_replaced] = self.mask_token_id
        # exam_value_mask[indices_replaced] = 0

        exam_outputs = self.get_exam_outputs(exam_input_values, exam_value_mask)
        fusion_outputs, fusion_features = self.get_fusion_features(input_ids, attention_mask, exam_outputs, exam_value_mask)
        diag_outputs, diag_features = self.get_diag_features(diag_input_ids, diag_attention_mask)
        drug_outputs, drug_features = self.get_drug_features(drug_input_ids, drug_attention_mask)

        note_embeds = fusion_features / fusion_features.norm(p=2, dim=-1, keepdim=True)
        diag_embeds = diag_features / diag_features.norm(p=2, dim=-1, keepdim=True)
        drug_embeds = drug_features / drug_features.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_note_0 = torch.matmul(diag_embeds, note_embeds.t()) * logit_scale[0]
        logits_per_note_1 = torch.matmul(drug_embeds, note_embeds.t()) * logit_scale[1]
        # logits_out = torch.matmul(drug_embeds, diag_embeds.t()) * logit_scale[2]
        # logits_per_diag = logits_per_note.t()



        loss = None
        if return_loss:
            loss = clip_loss(logits_per_note_0) + clip_loss(logits_per_note_1)

        if exam_input_labels is not None:
            exam_output_aug = fusion_outputs[:, :exam_outputs.shape[1]]
            prediction_scores = self.exam_mlm_head(exam_output_aug)
            loss_fct = CrossEntropyLoss(ignore_index=self.mask_token_id)
            masked_exam_loss = loss_fct(prediction_scores.view(-1, self.exam_num), exam_input_labels.view(-1).long())
            loss = loss + masked_exam_loss * logit_scale[2] * 0.5

        if not return_dict:
            output = (logits_per_note_0, diag_embeds, note_embeds, diag_features, fusion_features)
            return ((loss,) + output) if loss is not None else output

        return MultiOutput(
            loss=loss,
            logits_per_note=logits_per_note_0,
            note_embeds=note_embeds,
            diag_embeds=diag_embeds,
            drug_embeds=drug_embeds,
            note_model_output=fusion_outputs,
            diag_model_output=diag_outputs,
            drug_model_output=drug_outputs,
        )
