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
y2_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service', 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort', 'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation', 'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value', 'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']
domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description', 'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name', 'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city', 'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number', 'condition_description', 'condition_temperature']
domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
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