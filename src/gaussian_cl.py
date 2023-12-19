"""
@Author : Fendi Zhang <fendizh001@gmail.com>
@Start-Date : 2022-11-15
@Filename : gaussian_cl.py'
@Framework : Pytorch
"""

import torch
from torch import nn
from torch.nn import functional as F
from src.function import calculate_KL_or_euclidean
from transformers import AutoModel, AutoConfig

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

"""
BertConfig {
  "_name_or_path": "bert-base-uncased",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.3,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.25.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
"""
# embedding_dim

# coach_gcl: gold_labels
# ([0, 0, 0, 0, 69, 0, 41, 0, 0, 0, 0, 63], [0, 15, 16, 16, 16], [0, 0, 0, 39, 40, 40, 40, 0, 65, 0, 66],
#  [0, 35, 19, 65, 0, 0, 66, 67], [0, 0, 35, 57, 65, 0, 66], [0, 0, 0, 0, 0, 0, 17, 18, 0, 63, 0, 61, 62, 62],
#  [0, 0, 57, 0, 39, 40, 40, 40], [0, 0, 68, 3, 0, 23], [0, 0, 0, 0, 70, 0, 41, 0, 61],
#  [0, 0, 68, 3, 4, 0, 51, 52, 52, 0, 23, 24], [0, 35, 19, 65, 67], [0, 0, 57, 0, 9, 10, 10],
#  [0, 0, 0, 0, 0, 57, 0, 39, 40, 40, 40, 40, 40], [0, 0, 0, 39, 40, 40, 40], [0, 0, 0, 0, 39, 40, 40, 57],
#  [0, 0, 0, 49, 45, 46, 0, 0, 0, 9, 10, 10])

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01).to(device)
        nn.init.zeros_(m.bias).to(device)

class GaussianCL(nn.Module):
    def __init__(self, config, model_args):
        super(GaussianCL, self).__init__()
        self.model_args = model_args
        self.dropout = nn.Dropout(self.model_args.dropout_rate).to(device)
        self.hidden_size = config.hidden_size
        self.embedding_dim = self.model_args.embedding_dim
        self.projection = nn.Sequential(nn.Linear(self.hidden_size, self.embedding_dim)).to(device)
        self.output_embedding_mu = nn.Sequential(nn.LeakyReLU(), nn.Linear(self.hidden_size,
                                                                      self.embedding_dim)).to(device)  # set self.embedding_dim = 32 for trial
        self.output_embedding_sigma = nn.Sequential(nn.LeakyReLU(), nn.Linear(self.hidden_size, self.embedding_dim)).to(device)

        self.output_embedding_mu.apply(init_normal).to(device)
        self.output_embedding_sigma.apply(init_normal).to(device)

    def forward(
            self,
            sequence_outputs=None,
            attention_mask=None,
            gold_labels=None,
            loss_type=None
    ):
        """
        Inputs:
           domains: domain list for each sample (bsz,)
           hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
           lengths:each sample length of each batch (bsz, seq_len)
           binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
           final_golds: used only in the training mode (bsz, seq_len)
           attenation_mask:
           loss_type: euclidean or KL loss type
        Outputs:
           KL_loss: KL loss of gaussian embedding contrastive between hidden layers of all utterances themselves in each batch
           (loss), (output_mus, output_sigmas, ), (hidden_layers)
        """
        sequence_outputs = sequence_outputs.to(device)
        utterance_sequence_features = self.dropout(sequence_outputs)
        original_embedding_mu = ((self.output_embedding_mu((utterance_sequence_features))))
        # https://0809zheng.github.io/2020/03/01/activation.html
        # F.celu() || F.elu() || F.gelu() || F.selu
        original_embedding_sigma = (F.elu(self.output_embedding_sigma((utterance_sequence_features)))) + 1 + 1e-16

        outputs = (original_embedding_mu, original_embedding_sigma,) + (utterance_sequence_features,)

        loss = calculate_KL_or_euclidean(self, original_embedding_mu, original_embedding_sigma, attention_mask,
                                             gold_labels, loss_type=loss_type, model_args=self.model_args)
        outputs = (loss,) + outputs

        return outputs  # (loss), (output_mus, output_sigmas, ), (utterance_sequence_features)






















































































