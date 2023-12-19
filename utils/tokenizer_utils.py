"""
@Author : Fendi Zhang <fendizh001@gmail.com>
@Start-Date : 2022-11-15
@Filename : tokenizer_utils.py
@Framework : Pytorch
@Copyright (C) 2022. All Rights Reserved.
"""
import os
from transformers import AutoTokenizer
import spacy
import numpy as np


class Bert4Tokenizer:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        # https: // github.com / huggingface / transformers / issues / 5587
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name, use_fast=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
        print("vocab_size: {}".format(self.tokenizer.vocab_size))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', max_seq_len=None):
        out = self.tokenizer(text)
        return out
        # tokens = self.tokenizer.tokenize(text)
        # print("Bert_tokenize_tokens: {}".format(tokens))
        # print("len(Bert_tokenize_tokens): {}".format(len(tokens)))
        # sequence = self.tokenizer.convert_tokens_to_ids(tokens)
        # if len(sequence) == 0:
        #     sequence = [0]
        # if reverse:
        #     sequence = sequence[::-1]
        # if max_seq_len is None:
        #     max_seq_len = self.max_seq_len
        # return sequence
        # return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)


class Spacy4Tokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def text_to_token(self, text, reverse=False):
        if reverse:
            text = text[::-1]
        doc = self.nlp(text)
        return [token.text for token in doc]

    def text_to_dep(self, text, reverse=False):
        if reverse:
            text = text[::-1]
        doc = self.nlp(text)
        return [token.dep_ for token in doc]

    def text_to_pos(self, text, reverse=False):
        if reverse:
            text = text[::-1]
        doc = self.nlp(text)
        return [token.tag_ for token in doc]


if __name__ == "__main__":
    # tokenize category
    # sentence description

    sequence = "Justin Drew Bieber is a Canadian singer, songwriter, and actor."
    pretrained_bert_name = "bert-base-uncased"
    bt = Bert4Tokenizer(128, pretrained_bert_name)
    st = Spacy4Tokenizer()
    # output = bt.tokenizer(sequence)
    # print("len(output): ", len(output))
    # print("output: ", output)

    # vocab_size: 30522
    # Bert_tokenize_tokens: ['[CLS]', 'justin', 'drew', 'bi', '##eber', 'is', 'a', 'canadian', 'singer', ',',
    #                        'songwriter', ',', 'and', 'actor', '.', '[SEP]']
    # [101, 6796, 3881, 12170, 22669, 2003, 1037, 3010, 3220, 1010, 6009, 1010, 1998, 3364, 1012, 102]
    # deps = ['compound', 'compound', 'nsubj', 'ROOT', 'det', 'amod', 'attr', 'punct', 'conj', 'punct', 'cc', 'conj', 'punct']
    # ['NNP', 'NNP', 'NNP', 'VBZ', 'DT', 'JJ', 'NN', ',', 'NN', ',', 'CC', 'NN', '.']
    # ['Justin', 'Drew', 'Bieber', 'is', 'a', 'Canadian', 'singer', ',', 'songwriter', ',', 'and', 'actor', '.']
    # Bert_tokenize_tokens: ['[CLS]', 'compound', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'compound', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'ns', '##ub', '##j', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'root', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'det', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'am', '##od', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'at', '##tr', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'pun', '##ct', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'con', '##j', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'pun', '##ct', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'cc', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'con', '##j', '[SEP]']
    # Bert_tokenize_tokens: ['[CLS]', 'pun', '##ct', '[SEP]']
    # [[101, 7328, 102], [101, 7328, 102], [101, 24978, 12083, 3501, 102], [101, 7117, 102], [101, 20010, 102],
    #  [101, 2572, 7716, 102], [101, 2012, 16344, 102], [101, 26136, 6593, 102], [101, 9530, 3501, 102],
    #  [101, 26136, 6593, 102], [101, 10507, 102], [101, 9530, 3501, 102], [101, 26136, 6593, 102]]

    # print(bt.text_to_sequence(sequence))
    # print(st.text_to_dep(sequence))
    # print(st.text_to_pos(sequence))
    # print(st.text_to_token(sequence))
    # print([bt.text_to_sequence(token) for token in deps])
    print("utter: ", 'find movie schedules for imax corporation')
    print('find movie schedules for imax corporation'.split())

    print(st.text_to_dep('find movie schedules for imax corporation'))
    print(st.text_to_pos('find movie schedules for imax corporation'))

