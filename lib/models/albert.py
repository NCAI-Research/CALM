# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ALBERT modules that do not hog your GPU memory """

import torch
import torch.nn as nn
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import PreTrainedModel
from transformers.models.albert.modeling_albert import (
    ALBERT_START_DOCSTRING, AlbertForPreTraining,
    AlbertForSequenceClassification, AlbertForTokenClassification,
    AlbertMLMHead, AlbertModel, AlbertSOPHead)
from transformers.utils import logging

from lib.models.transformer import LeanTransformer, LeanTransformerConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LeanAlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


class LeanAlbertConfig(LeanTransformerConfig):
    def __init__(
        self,
        *args,
        classifier_dropout_prob: float = 0.1,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs
    ):
        super().__init__(
            *args,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
        )
        self.classifier_dropout_prob = classifier_dropout_prob
        self.type_vocab_size = type_vocab_size


class LeanAlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: LeanTransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)

        self.token_type_embeddings = config.get_token_type_embeddings()
        self.position_embeddings = config.get_input_position_embeddings()

        self.layernorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.position_embeddings is not None:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@add_start_docstrings(
    "The bare ALBERT-based LeanTransformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class LeanAlbertModel(AlbertModel):
    config_class = LeanAlbertConfig

    def __init__(self, config: config_class, add_pooling_layer=True):
        PreTrainedModel.__init__(self, config)

        self.config = config
        self.embeddings = LeanAlbertEmbeddings(config)
        self.encoder = LeanTransformer(config)

        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()


class GradientCheckpointingMixin:
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool):
        if isinstance(module, LeanTransformer):
            module.gradient_checkpointing = value


class LeanAlbertForPreTraining(GradientCheckpointingMixin, AlbertForPreTraining, PreTrainedModel):
    config_class = LeanAlbertConfig
    base_model_prefix = "lean_albert"

    def __init__(self, config: config_class):
        PreTrainedModel.__init__(self, config)

        self.albert = LeanAlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.albert.embeddings.word_embeddings = new_embeddings

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LeanAlbertForTokenClassification(AlbertForTokenClassification, PreTrainedModel, GradientCheckpointingMixin):
    config_class = LeanAlbertConfig
    base_model_prefix = "lean_albert"

    def __init__(self, config: config_class):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = LeanAlbertModel(config, add_pooling_layer=False)
        classifier_dropout_prob = (
            config.classifier_dropout_prob if config.classifier_dropout_prob is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()


class LeanAlbertForSequenceClassification(AlbertForSequenceClassification, PreTrainedModel, GradientCheckpointingMixin):
    config_class = LeanAlbertConfig
    base_model_prefix = "lean_albert"

    def __init__(self, config: config_class):
        PreTrainedModel.__init__(self,config)
        self.num_labels = config.num_labels
        self.config = config
        self.albert = LeanAlbertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

