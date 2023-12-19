# from preprocess.gen_embeddings_for_slu import domain2slot
import time
from utils.tokenizer_utils import Spacy4Tokenizer

# import logging


# logger = logging.getLogger()

y1_set = ["O", "B", "I"]
# len(y2_set)) =》 72
y2_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 'I-geographic_poi',
          'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name',
          'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type',
          'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service',
          'I-service', 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish',
          'I-served_dish', 'B-genre', 'I-genre', 'B-current_location', 'I-current_location', 'B-object_select',
          'I-object_select', 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort',
          'I-sort', 'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type',
          'B-spatial_relation', 'I-spatial_relation', 'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name',
          'I-entity_name', 'B-object_type', 'I-object_type', 'B-playlist_owner', 'I-playlist_owner', 'B-timeRange',
          'I-timeRange', 'B-city', 'I-city', 'B-rating_value', 'B-best_rating', 'B-rating_unit', 'B-year',
          'B-party_size_number', 'B-condition_description', 'B-condition_temperature']
# len(y2_coarse_set) =》 40
y2_coarse_set = ['O', 'party_size_number', 'rating_value', 'object_part_of_series_type', 'restaurant_type', 'rating_unit', 'poi', 'city', 'facility', 'movie_name', 'sort', 'restaurant_name', 'object_select', 'service', 'entity_name', 'condition_description', 'current_location', 'track', 'playlist_owner', 'country', 'state', 'condition_temperature', 'object_type', 'spatial_relation', 'cuisine', 'object_location_type', 'timeRange', 'best_rating', 'party_size_description', 'object_name', 'artist', 'geographic_poi', 'year', 'served_dish', 'album', 'location_name', 'playlist', 'music_item', 'genre', 'movie_type']

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type',
                       'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state',
                       'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi',
                   'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album', 'sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type',
                 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type', 'object_type', 'location_name',
                             'spatial_relation', 'movie_name']
}

coarse2id = {'O': 0, 'party_size_number': 1, 'rating_value': 2, 'object_part_of_series_type': 3, 'restaurant_type': 4, 'rating_unit': 5, 'poi': 6, 'city': 7, 'facility': 8, 'movie_name': 9, 'sort': 10, 'restaurant_name': 11, 'object_select': 12, 'service': 13, 'entity_name': 14, 'condition_description': 15, 'current_location': 16, 'track': 17, 'playlist_owner': 18, 'country': 19, 'state': 20, 'condition_temperature': 21, 'object_type': 22, 'spatial_relation': 23, 'cuisine': 24, 'object_location_type': 25, 'timeRange': 26, 'best_rating': 27, 'party_size_description': 28, 'object_name': 29, 'artist': 30, 'geographic_poi': 31, 'year': 32, 'served_dish': 33, 'album': 34, 'location_name': 35, 'playlist': 36, 'music_item': 37, 'genre': 38, 'movie_type': 39}

fine2id = {'O': 0, 'B-playlist': 1, 'I-playlist': 2, 'B-music_item': 3, 'I-music_item': 4, 'B-geographic_poi': 5, 'I-geographic_poi': 6, 'B-facility': 7, 'I-facility': 8, 'B-movie_name': 9, 'I-movie_name': 10, 'B-location_name': 11, 'I-location_name': 12, 'B-restaurant_name': 13, 'I-restaurant_name': 14, 'B-track': 15, 'I-track': 16, 'B-restaurant_type': 17, 'I-restaurant_type': 18, 'B-object_part_of_series_type': 19, 'I-object_part_of_series_type': 20, 'B-country': 21, 'I-country': 22, 'B-service': 23, 'I-service': 24, 'B-poi': 25, 'I-poi': 26, 'B-party_size_description': 27, 'I-party_size_description': 28, 'B-served_dish': 29, 'I-served_dish': 30, 'B-genre': 31, 'I-genre': 32, 'B-current_location': 33, 'I-current_location': 34, 'B-object_select': 35, 'I-object_select': 36, 'B-album': 37, 'I-album': 38, 'B-object_name': 39, 'I-object_name': 40, 'B-state': 41, 'I-state': 42, 'B-sort': 43, 'I-sort': 44, 'B-object_location_type': 45, 'I-object_location_type': 46, 'B-movie_type': 47, 'I-movie_type': 48, 'B-spatial_relation': 49, 'I-spatial_relation': 50, 'B-artist': 51, 'I-artist': 52, 'B-cuisine': 53, 'I-cuisine': 54, 'B-entity_name': 55, 'I-entity_name': 56, 'B-object_type': 57, 'I-object_type': 58, 'B-playlist_owner': 59, 'I-playlist_owner': 60, 'B-timeRange': 61, 'I-timeRange': 62, 'B-city': 63, 'I-city': 64, 'B-rating_value': 65, 'B-best_rating': 66, 'B-rating_unit': 67, 'B-year': 68, 'B-party_size_number': 69, 'B-condition_description': 70, 'B-condition_temperature': 71}


SLOT_PAD = 0
PAD_INDEX = 0
UNK_INDEX = 1


def read_file(filepath, domain=None, use_dep=False, use_pos=False):
    domain_list, label_list, utter_list, y_list, utter_dep_list, utter_pos_list, sen_coarse_label_list, sen_fine_label_list = [], [], [], [], [], [], [], []
    # domain_list, label_list, utter_list, y_list, utter_dep_list, utter_pos_list = [], [], [], [], [], []
    # domain_list, label_list, utter_list, y_list = [], [], [], []
    '''
    domain_list: lists of domain
    label_list: lists of slot label
    utter_list: lists of query
    y1_list: lists of BIO labels w/o labels
    sen_coarse_label_list: utterance level coarse slot label
    sen_fine_label_list: utterance level fine slot label
    '''
    max_length = 0
    data_size = 0

    if use_dep and use_pos:
        # utter_dep_list = []
        # utter_pos_list = []
        st_dep = Spacy4Tokenizer()
        st_pos = st_dep
    #     st_pos = Spacy4Tokenizer()
    elif use_dep:
        # utter_dep_list = []
        st_dep = Spacy4Tokenizer()
        # print("spacy dep ready!")
    elif use_pos:
        # utter_pos_list = []
        st_pos = Spacy4Tokenizer()
        # print("spacy pos ready!")

    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            data_size += 1
            line = line.strip()  # query \t BIO-labels
            splits = line.split("\t")  # split query and labels
            utter = splits[0]
            if use_dep and use_pos:
                utter_dep = " ".join(st_dep.text_to_dep(utter))
                utter_pos = " ".join(st_pos.text_to_pos(utter))
            if use_dep:
                utter_dep = " ".join(st_dep.text_to_dep(utter))
            elif use_pos:
                utter_pos = " ".join(st_pos.text_to_pos(utter))
            tokens = splits[0].split()
            l2_list = splits[1].split()  # O B-LB1 I-LB1 ....
            if max_length < len(tokens):
                max_length = len(tokens)
            sen_fine_list = []
            sen_fine_list = l2_list
            sen_coarse_list=[]
            #utterance coarse label
            for i, l in enumerate(l2_list):
                if "B" in l:
                    tag, slot = l.split('-')
                    sen_coarse_list.append(slot)
                elif "I" in l:
                    tag, slot = l.split('-')
                    sen_coarse_list.append(slot)
                else:
                    sen_coarse_list.append('O')

            sen_fine_list_bin = [fine2id[c] for c in sen_fine_list]
            sen_coarse_list_bin = [coarse2id[f] for f in sen_coarse_list]

            # for each label, make B/I/O labeled target 
            BIO_with_slot_dict = {}
            for i, l in enumerate(l2_list):
                if "B" in l:
                    tag, slot = l.split('-')
                    BIO_with_slot_dict[slot] = [0 for _ in range(len(l2_list))]
                    BIO_with_slot_dict[slot][i] = 1
                elif "I" in l:
                    tag, slot = l.split('-')
                    BIO_with_slot_dict[slot][i] = 2

            #data augmentation=> every slot add two negative samples
            # negative_samples_num_limit = 0
            #
            # for slot in domain2slot[domain]:
            #     # setting up negative sample numbers to 2
            #     if negative_samples_num_limit >= 2:
            #         break
            #     if slot not in BIO_with_slot_dict.keys(): # some slots may not be in utterance: just add list of "O"s
            #         negative_samples_num_limit += 1
            #         BIO_with_slot_dict[slot] = [0 for _ in range(len(l2_list))]
            # print("negative_samples_num_limit: {}".format(negative_samples_num_limit))

            # simulate real inference scenario
            for slot in domain2slot[domain]:
                if slot not in BIO_with_slot_dict.keys(): # some slots may not be in utterance: just add list of "O"s
                    BIO_with_slot_dict[slot] = [0 for _ in range(len(l2_list))]

            slot_list = list(BIO_with_slot_dict.keys())
            BIO_list = list(BIO_with_slot_dict.values())
            # print(BIO_with_slot_dict)

            domain_list.extend([domain for _ in range(len(slot_list))])
            utter_list.extend([utter for _ in range(len(slot_list))])
            sen_coarse_label_list.extend([sen_coarse_list_bin for _ in range(len(slot_list))])
            sen_fine_label_list.extend([sen_fine_list_bin for _ in range(len(slot_list))])

            if use_dep and use_pos:
                utter_dep_list.extend([utter_dep for _ in range(len(slot_list))])
                utter_pos_list.extend([utter_pos for _ in range(len(slot_list))])
            elif use_dep:
                utter_dep_list.extend([utter_dep for _ in range(len(slot_list))])
            elif use_pos:
                utter_pos_list.extend([utter_pos for _ in range(len(slot_list))])
            label_list.extend(slot_list)
            y_list.extend(BIO_list)
            # sen_coarse_label_list.extend(sen_coarse_list_bin)
            # sen_fine_label_list.extend(sen_fine_list_bin)

    # if use_dep and use_pos:
    #     data_dict = {"domain": domain_list, "label": label_list, "utter": utter_list, "y": y_list, "utter_dep": utter_dep_list, "utter_pos": utter_pos_list}
    # elif use_dep:
    #     data_dict = {"domain": domain_list, "label": label_list, "utter": utter_list, "y": y_list,
    #                  "utter_dep": utter_dep_list}
    # elif use_pos:
    #     data_dict = {"domain": domain_list, "label": label_list, "utter": utter_list, "y": y_list,
    #                  "utter_pos": utter_pos_list}
    # else:
    #     data_dict = {"domain": domain_list, "label": label_list, "utter": utter_list, "y": y_list}

    data_dict = {"domain": domain_list, "label": label_list, "utter": utter_list, "y": y_list,
                 "utter_dep": utter_dep_list, "utter_pos": utter_pos_list, "gold_coarse_label": sen_coarse_label_list, "gold_fine_label": sen_fine_label_list}
    return data_dict, max_length, int(data_size)


def data_binarize(data):
    data_bin = {"domain": [], "label": [], "utter": [], "y": []}
    for domain_list, label_list, utter_list, y_list in zip(data['domain'], data['label'], data['utter'], data['y']):
        y_bin = []
        for y in y_list:
            y_bin.append(y1_set.index(y))

        data_bin['domain'].append(domain_list)
        data_bin['label'].append(label_list)
        data_bin['utter'].append(utter_list)
        data_bin['y'].append(y_bin)

    return data_bin


def datareader(data_path, use_dep, use_pos):
    # logger.info("Loading and processing data ...")

    data = {"AddToPlaylist": {}, "BookRestaurant": {}, "GetWeather": {}, "PlayMusic": {}, "RateBook": {},
            "SearchCreativeWork": {}, "SearchScreeningEvent": {}}
    max_length = {"AddToPlaylist": 0, "BookRestaurant": 0, "GetWeather": 0, "PlayMusic": 0, "RateBook": 0,
                  "SearchCreativeWork": 0, "SearchScreeningEvent": 0}

    data_size = {"AddToPlaylist": 0, "BookRestaurant": 0, "GetWeather": 0, "PlayMusic": 0, "RateBook": 0,
                  "SearchCreativeWork": 0, "SearchScreeningEvent": 0}

    # load data
    data['AddToPlaylist'], max_length['AddToPlaylist'], data_size['AddToPlaylist'] = read_file(f"{data_path}/AddToPlaylist/AddToPlaylist.txt",
                                                                   domain="AddToPlaylist", use_dep=use_dep, use_pos=use_pos)
    data['BookRestaurant'], max_length['BookRestaurant'], data_size['BookRestaurant'] = read_file(f"{data_path}/BookRestaurant/BookRestaurant.txt",
                                                                     domain="BookRestaurant", use_dep=use_dep, use_pos=use_pos)
    data['GetWeather'], max_length['GetWeather'], data_size['GetWeather'] = read_file(f"{data_path}/GetWeather/GetWeather.txt",
                                                             domain="GetWeather", use_dep=use_dep, use_pos=use_pos)
    data['PlayMusic'], max_length['PlayMusic'], data_size['PlayMusic'] = read_file(f"{data_path}/PlayMusic/PlayMusic.txt", domain="PlayMusic", use_dep=use_dep, use_pos=use_pos)
    data['RateBook'], max_length['RateBook'], data_size['RateBook'] = read_file(f"{data_path}/RateBook/RateBook.txt", domain="RateBook",use_dep=use_dep, use_pos=use_pos)
    data['SearchCreativeWork'], max_length['SearchCreativeWork'], data_size['SearchCreativeWork'] = read_file(
        f"{data_path}/SearchCreativeWork/SearchCreativeWork.txt", domain="SearchCreativeWork", use_dep=use_dep, use_pos=use_pos)
    data['SearchScreeningEvent'], max_length['SearchScreeningEvent'], data_size['SearchScreeningEvent'] = read_file(
        f"{data_path}/SearchScreeningEvent/SearchScreeningEvent.txt", domain="SearchScreeningEvent", use_dep=use_dep, use_pos=use_pos)

    # # unseen
    # data['AddToPlaylist_unseen'], max_length['AddToPlaylist_unseen'], data_size['AddToPlaylist_unseen'] = read_file(
    #     f"{data_path}/AddToPlaylist/unseen_slots.txt",
    #     domain="AddToPlaylist", use_dep=use_dep, use_pos=use_pos)
    # data['BookRestaurant_unseen'], max_length['BookRestaurant_unseen'], data_size['BookRestaurant_unseen'] = read_file(
    #     f"{data_path}/BookRestaurant/unseen_slots.txt",
    #     domain="BookRestaurant", use_dep=use_dep, use_pos=use_pos)
    # data['GetWeather_unseen'], max_length['GetWeather_unseen'], data_size['GetWeather_unseen'] = read_file(
    #     f"{data_path}/GetWeather/unseen_slots.txt",
    #     domain="GetWeather", use_dep=use_dep, use_pos=use_pos)
    # data['PlayMusic_unseen'], max_length['PlayMusic_unseen'], data_size['PlayMusic_unseen'] = read_file(
    #     f"{data_path}/PlayMusic/unseen_slots.txt", domain="PlayMusic", use_dep=use_dep, use_pos=use_pos)
    # data['RateBook_unseen'], max_length['RateBook_unseen'], data_size['RateBook_unseen'] = read_file(f"{data_path}/RateBook/unseen_slots.txt",
    #                                                                             domain="RateBook", use_dep=use_dep,
    #                                                                             use_pos=use_pos)
    # data['SearchCreativeWork_unseen'], max_length['SearchCreativeWork_unseen'], data_size['SearchCreativeWork_unseen'] = read_file(
    #     f"{data_path}/SearchCreativeWork/unseen_slots.txt", domain="SearchCreativeWork", use_dep=use_dep,
    #     use_pos=use_pos)
    # data['SearchScreeningEvent_unseen'], max_length['SearchScreeningEvent_unseen'], data_size['SearchScreeningEvent_unseen'] = read_file(
    #     f"{data_path}/SearchScreeningEvent/unseen_slots.txt", domain="SearchScreeningEvent", use_dep=use_dep,
    #     use_pos=use_pos)

    # # seen
    # data['AddToPlaylist_seen'], max_length['AddToPlaylist_seen'], data_size['AddToPlaylist_seen'] = read_file(
    #     f"{data_path}/AddToPlaylist/seen_slots.txt",
    #     domain="AddToPlaylist", use_dep=use_dep, use_pos=use_pos)
    # data['BookRestaurant_seen'], max_length['BookRestaurant_seen'], data_size['BookRestaurant_seen'] = read_file(
    #     f"{data_path}/BookRestaurant/seen_slots.txt",
    #     domain="BookRestaurant", use_dep=use_dep, use_pos=use_pos)
    # data['GetWeather_seen'], max_length['GetWeather_seen'], data_size['GetWeather_seen'] = read_file(
    #     f"{data_path}/GetWeather/seen_slots.txt",
    #     domain="GetWeather", use_dep=use_dep, use_pos=use_pos)
    # data['PlayMusic_seen'], max_length['PlayMusic_seen'], data_size['PlayMusic_seen'] = read_file(
    #     f"{data_path}/PlayMusic/seen_slots.txt", domain="PlayMusic", use_dep=use_dep, use_pos=use_pos)
    # data['RateBook_seen'], max_length['RateBook_seen'], data_size['RateBook_seen'] = read_file(f"{data_path}/RateBook/seen_slots.txt",
    #                                                                             domain="RateBook", use_dep=use_dep,
    #                                                                             use_pos=use_pos)
    # data['SearchCreativeWork_seen'], max_length['SearchCreativeWork_seen'], data_size['SearchCreativeWork_seen'] = read_file(
    #     f"{data_path}/SearchCreativeWork/seen_slots.txt", domain="SearchCreativeWork", use_dep=use_dep,
    #     use_pos=use_pos)
    # data['SearchScreeningEvent_seen'], max_length['SearchScreeningEvent_seen'], data_size['SearchScreeningEvent_seen'] = read_file(
    #     f"{data_path}/SearchScreeningEvent/seen_slots.txt", domain="SearchScreeningEvent", use_dep=use_dep,
    #     use_pos=use_pos)

    # data_atp, max_length['AddToPlaylist'] = read_file(f"{data_path}/AddToPlaylist/AddToPlaylist.txt", domain="AddToPlaylist")
    # data_br, max_length['BookRestaurant'] = read_file(f"{data_path}/BookRestaurant/BookRestaurant.txt", domain="BookRestaurant")
    # data_gw, max_length['GetWeather'] = read_file(f"{data_path}/GetWeather/GetWeather.txt", domain="GetWeather")
    # data_pm, max_length['PlayMusic'] = read_file(f"{data_path}/PlayMusic/PlayMusic.txt", domain="PlayMusic")
    # data_rb, max_length['RateBook'] = read_file(f"{data_path}/RateBook/RateBook.txt", domain="RateBook")
    # data_scw, max_length['SearchCreativeWork'] = read_file(f"{data_path}/SearchCreativeWork/SearchCreativeWork.txt", domain="SearchCreativeWork")
    # data_sse, max_length['SearchScreeningEvent'] = read_file(f"{data_path}/SearchScreeningEvent/SearchScreeningEvent.txt", domain="SearchScreeningEvent")

    # data["AddToPlaylist"] =  data_binarize(data_atp)
    # data["BookRestaurant"] =  data_binarize(data_br)
    # data["GetWeather"] =  data_binarize(data_gw)
    # data["PlayMusic"] =  data_binarize(data_pm)
    # data["RateBook"] =  data_binarize(data_rb)
    # data["SearchCreativeWork"] =  data_binarize(data_scw)
    # data["SearchScreeningEvent"] =  data_binarize(data_sse)

    # print(max_length)
    # print(data_size)
    return data, max(max_length.values()), data_size


if __name__ == "__main__":
    # print(len(y2_set))
    # print(len(y2_coarse_set))
    coarse2id = {}
    fine2id={}
    for i, label in enumerate(y2_coarse_set):
        coarse2id[label] = i

    for i, label in enumerate(y2_set):
        fine2id[label] = i

    print(coarse2id)
    print(fine2id)
    # y2_coarse_set = []
    # for v in domain2slot.values():
    #     for s in v:
    #         y2_coarse_set.append(s)
    # print(y2_coarse_set)
    # y2_coarse_set = list(set(y2_coarse_set))
    # print(y2_coarse_set)
    # print(len(y2_coarse_set))
