"""
@Author : Fendi Zhang <fendizh001@gmail.com>
@Start-Date : 2022-11-15
@Filename : model.py'
@Framework : Pytorch
"""

from transformers import AutoConfig, AutoModel
from src.modules import CRF
import torch
from torch import nn
from src.gaussian_cl import GaussianCL

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class TripletTagger(nn.Module):
    def __init__(self, bert_hidden_dim, num_binslot=3):
        super(TripletTagger, self).__init__()

        self.num_binslot = num_binslot
        self.hidden_dim = bert_hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)

    def forward(self, inputs, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            prediction: logits of predictions
            crf_loss: loss of crf
        """
        prediction = self.linear(inputs)
        crf_loss = self.crf_layer.loss(prediction, y)
        return prediction, crf_loss

    def crf_decode(self, logits):
        """
        crf decode
        logits to labeling (0/1/2 == O/B/I)
        Input:
            logits: (bsz, max_seq_len, num_entity)
        Output:
            pred: (bsz, max_seq_len)
        """
        return torch.argmax(logits, dim=2)


class ZSSFModel(nn.Module):
    def __init__(self, model_args, training_args):
        super(ZSSFModel, self).__init__()
        # model_args
        self.model_args = model_args
        # training_args
        self.training_args = training_args
        # hyperparameter
        self.num_tags = self.model_args.num_tags

        # model
        self.model_name = self.model_args.model_name_or_path
        self.model_config = AutoConfig.from_pretrained(self.model_name, hidden_dropout_prob=self.model_args.dropout_rate)

        self.utterance_pretrained_model = AutoModel.from_pretrained(self.model_name, config=self.model_config)
        self.dropout = nn.Dropout(p=self.model_args.dropout_rate)
        self.classifier = nn.Linear(self.utterance_pretrained_model.config.hidden_size, self.num_tags)
        self.crf = CRF(self.num_tags)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.config = self.model_config

    def forward(self, u_inputs):
        """
        parameters
        ----------
        u_inputs: consist of (tokenized (slot_desc, utterance) pairs, (gold) bio labels)
        u_inputs(add dependency parser): consist of (tokenized (slot_desc, utterance_dep, utterance) pairs, (gold) bio labels)
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': encoded['token_type_ids'],
        'labels': new_labels,
        'gold_coarse_labels': new_gold_coarse_labels,
        'gold_fine_labels' : new_gold_fine_labels
        """
        utterance_outputs = self.utterance_pretrained_model(
            u_inputs['input_ids'],
            u_inputs['attention_mask'],
            u_inputs['token_type_ids']
        )
        utterance_sequence_outputs = utterance_outputs.last_hidden_state
        utterance_sequence_outputs = utterance_sequence_outputs.to(device)
        utterance_cls_output = utterance_sequence_outputs[:, 0, :]

        # BIO classification
        logits = self.classifier(utterance_sequence_outputs)
        crf_loss = self.crf.loss(logits, u_inputs['labels'])

        # gaussian contrastive learning

        # gcl_training_loss
        # sequence_outputs = None,
        # attention_mask = None,
        # gold_labels = None,
        # loss_type = None
        coarse_gold_labels = u_inputs['gold_coarse_labels']
        fine_gold_labels = u_inputs['gold_fine_labels']
        if self.model_args.use_gaussian_cl and self.model_args.use_coarse_gold_label and self.model_args.use_fine_gold_label:
            gcl = GaussianCL(self.config, self.model_args).to(device)
            if coarse_gold_labels != [] and fine_gold_labels != []:
                losses = []
                coarse_outputs = gcl(sequence_outputs=utterance_sequence_outputs, attention_mask=u_inputs['attention_mask'], gold_labels=coarse_gold_labels,
                loss_type=self.training_args.gcl_training_loss)
                coarse_loss = coarse_outputs[0]

                # coarse_loss = coarse_loss.mean()
                # optim.zero_grad()
                # # coarse_loss.backward()
                # coarse_loss.backward(retain_graph=True)
                # losses.append(coarse_loss.detach().cpu().item())
                #
                # optim.step()
                # scheduler.step()

                # print("Coarse_LOSS: {:.4f} ".format(losses[-1])) # Coarse_LOSS: 7.913882

                fine_outputs = gcl(sequence_outputs=utterance_sequence_outputs, attention_mask=u_inputs['attention_mask'], gold_labels=fine_gold_labels,
                loss_type=self.training_args.gcl_training_loss)
                fine_loss = fine_outputs[0]

                total_outputs = (logits,) + (coarse_outputs[1:],) + fine_outputs[1:]
                # total_loss = self.model_args.alpha * crf_loss + self.model_args.beta * coarse_loss + self.model_args.gamma * fine_loss  # L2 regularization

                # hierachical gaussian contrastive learning setting
                # alpha = 0.9   gamma=0.1
                total_loss = self.model_args.alpha * crf_loss + self.model_args.gamma * fine_loss  # L2 regularization

                # return total_loss, total_outputs
                return crf_loss, coarse_loss, fine_loss, total_outputs

        elif self.model_args.use_gaussian_cl and self.model_args.use_coarse_gold_label and not self.model_args.use_fine_gold_label:
            gcl = GaussianCL(self.config, self.model_args).to(device)
            if coarse_gold_labels != []:
                coarse_outputs = gcl(sequence_outputs=utterance_sequence_outputs, attention_mask=u_inputs['attention_mask'], gold_labels=coarse_gold_labels,
                loss_type=self.training_args.gcl_training_loss)
                coarse_loss = coarse_outputs[0]

                total_outputs = (logits,) + (coarse_outputs[1:],)
                total_loss = self.model_args.alpha * crf_loss + self.model_args.beta * coarse_loss   # L2 regularization

                return total_loss, total_outputs

        elif self.model_args.use_gaussian_cl and not self.model_args.use_coarse_gold_label and self.model_args.use_fine_gold_label:
            gcl = GaussianCL(self.config, self.model_args).to(device)
            if fine_gold_labels != []:

                fine_outputs = gcl(sequence_outputs=utterance_sequence_outputs, attention_mask=u_inputs['attention_mask'],
                                   gold_labels=fine_gold_labels,
                                   loss_type=self.training_args.gcl_training_loss)
                fine_loss = fine_outputs[0]

                total_outputs = (logits,) + (fine_outputs[1:],)
                total_loss = self.model_args.alpha * crf_loss + self.model_args.gamma * fine_loss  # L2 regularization

                return total_loss, total_outputs

        else:
            return crf_loss, logits


