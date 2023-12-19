import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import pickle

y1_set = ["O", "B", "I"]



y2_set = ['O', 'B-hotel_bookstay', 'B-hotel_bookday', 'B-hotel_bookpeople', 'B-hotel_name', 'I-hotel_name', 'B-hotel_stars', 'I-hotel_stars', \
          'B-hotel_type', 'I-hotel_type', 'B-hotel_area', 'B-hotel_pricerange', 'B-restaurant_name', 'I-restaurant_name', 'B-taxi_destination', \
          'I-taxi_destination', 'B-taxi_departure', 'I-taxi_departure', 'B-taxi_arriveby', 'I-taxi_arriveby', 'B-restaurant_booktime', \
          'I-restaurant_booktime', 'B-hotel_parking', 'B-taxi_leaveat', 'I-taxi_leaveat', 'B-restaurant_food', 'I-restaurant_food', \
          'B-attraction_name', 'I-attraction_name', 'B-restaurant_bookpeople', 'B-restaurant_bookday', 'B-restaurant_pricerange', \
          'B-restaurant_area', 'B-train_arriveby', 'I-train_arriveby', 'B-hospital_department', 'I-hospital_department', \
          'B-train_bookpeople', 'B-train_day', 'B-train_leaveat', 'I-train_leaveat', 'B-train_departure', 'I-train_departure', 'B-train_destination', \
          'I-train_destination', 'B-attraction_area', 'B-attraction_type', 'I-attraction_type', 'B-hotel_internet']

domain_set = ["Book_Hotel", "Book_Restaurant", "Book_Train", "Find_Attraction", "Find_Hotel", "Find_Restaurant", "Find_Taxi", "Find_Train"]

slot_list = ['O', 'hotel_bookstay', 'hotel_bookday', 'hotel_bookpeople', 'hotel_name', 'hotel_stars', 'hotel_type', 'hotel_area', \
                 'hotel_pricerange', 'restaurant_name', 'taxi_destination', 'taxi_departure', 'taxi_arriveby', 'restaurant_booktime', \
                 'hotel_parking', 'taxi_leaveat', 'restaurant_food', 'attraction_name', 'restaurant_bookpeople', 'restaurant_bookday', \
                 'restaurant_pricerange', 'restaurant_area', 'train_arriveby', 'hospital_department', 'train_bookpeople', 'train_day', \
                 'train_leaveat', 'train_departure', 'train_destination', 'attraction_area', 'attraction_type', 'hotel_internet']


domain2slot = {'Book_Hotel': ['taxi_arriveby', 'restaurant_booktime', 'taxi_leaveat', 'hotel_parking', 'hotel_bookday', 'hotel_pricerange', 'attraction_name', 'hotel_name', 'hotel_type', 'hotel_stars', 'taxi_destination', 'hotel_area', 'hotel_bookpeople', 'restaurant_food', 'taxi_departure', 'restaurant_name', 'hotel_bookstay'], \
               'Book_Restaurant': ['restaurant_booktime', 'restaurant_area', 'restaurant_bookday', 'taxi_arriveby', 'taxi_leaveat', 'train_arriveby', 'hotel_name', 'taxi_destination', 'restaurant_food', 'hospital_department', 'restaurant_pricerange', 'taxi_departure', 'restaurant_name', 'restaurant_bookpeople'], \
               'Book_Train': ['train_arriveby', 'attraction_name', 'hotel_name', 'train_bookpeople', 'train_destination', 'train_departure', 'restaurant_food', 'train_day', 'train_leaveat', 'restaurant_name'], \
               'Find_Attraction': ['taxi_arriveby', 'taxi_leaveat', 'attraction_name', 'hotel_name', 'taxi_destination', 'restaurant_food', 'restaurant_name', 'hospital_department', 'taxi_departure', 'attraction_area', 'attraction_type'], \
               'Find_Hotel': ['taxi_arriveby', 'hotel_parking', 'hotel_pricerange', 'attraction_name', 'hotel_name', 'hotel_type', 'hotel_stars', 'taxi_destination', 'hotel_area', 'restaurant_food', 'hospital_department', 'taxi_departure', 'restaurant_name', 'hotel_internet'], \
               'Find_Restaurant': ['restaurant_area', 'restaurant_booktime', 'restaurant_bookday', 'attraction_name', 'hotel_name', 'taxi_destination', 'restaurant_food', 'hospital_department', 'restaurant_pricerange', 'taxi_departure', 'restaurant_name', 'restaurant_bookpeople'], \
               'Find_Taxi': ['taxi_arriveby', 'taxi_leaveat', 'restaurant_booktime', 'attraction_name', 'hotel_name', 'taxi_destination', 'hospital_department', 'taxi_departure'], \
               'Find_Train': ['restaurant_booktime', 'taxi_leaveat', 'train_arriveby', 'attraction_name', 'hotel_name', 'train_destination', 'train_bookpeople', 'taxi_destination', 'train_departure', 'restaurant_food', 'train_day', 'taxi_departure', 'train_leaveat', 'restaurant_name']
               }


def make_syn_data(slot_dict, template_list, num_aug=1):
    """
    합성 문장을 만드는 함수
    positive augmented data: 1
    negative augmented data: num_aug
    if num_aug == 0: positive augmented data == negative augmented data == 0

    input
    -------
    slot_dict: 슬롯 라벨 및 거기 해당하는 슬롯 예시들의 dictionary
    template_list: 슬롯에 해당하는 단어가 T-slot_label 로 대체된 문장들, 3개 문장의 list로 구성
    how_many: 몇 개 만들지

    output
    -------
    syn_sents: 합성된 문장들
    """
    syn_sents = []
    for i, template in enumerate(template_list):
        if i == 0 and num_aug > 0: 
            how_many = 1
        else: 
            how_many = num_aug

        for _ in range(how_many): # make augmented data replacing slot labels by slot exemplars 
            syn_sent = []
            for word in template.split():
                if "T-" in word:
                    slot_label = word.split('-')[1]
                    slot_exemplar_list = slot_dict[slot_label]
                    slot_exemplar = slot_exemplar_list[random.randint(0, len(slot_exemplar_list) - 1)] # get slot exemplar word
                    syn_sent.append(slot_exemplar)
                else:
                    syn_sent.append(word)

            syn_sent = " ".join(syn_sent)
            syn_sents.append(syn_sent)

    if len(syn_sents) > 0:
        return syn_sents
    else:
        return None


def update_dict(dict1, dict2):
    """
    update contents of dict1 with dict2 which value is list
    """
    assert type(dict1) is dict and type(dict2) is dict
    for key, value in dict2.items():
        if key in dict1.keys():
            dict1[key].extend(value)
        else:
            dict1[key] = value

    return dict1 


def log_params(json_dict, params):
    """
    read from params(list of parameters) and write on json_dict

    inputs
    ----------
    json_dict: dictionary, 
    params: list, list of params
    """
    necessary_params = ['dataset_name', 'target_domain', 'n_samples', 'num_aug_data', 'key_enc_data', 'num_key_enc_data', 'loss_key_ratio', 'dropout_rate', 'learning_rate', 'max_steps', 'eval_steps', 'early_stopping_patience', 'per_device_train_batch_size', 'warmup_steps']
    json_dict['parameters'] = {}

    for param in params:
        for param_keyword in necessary_params:
            if hasattr(param, param_keyword):
                json_dict['parameters'][param_keyword] = getattr(param, param_keyword)
    


def save_plot(title, xlabel, ylabel, file_path, data_y, data_x = None, dpi=500):
    plt.figure()
    if data_x is not None:
        plt.plot(data_x, data_y)
    else:
        plt.plot(data_y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path, dpi=dpi)
    plt.close()