# from preprocess.gen_embeddings_for_slu import domain2slot
import time
from utils.tokenizer_utils import Spacy4Tokenizer

# import logging


# logger = logging.getLogger()

y1_set = ["O", "B", "I"]
# len(y2_set)) =》 49
y2_set = ['O', 'B-hotel_bookstay', 'B-hotel_bookday', 'B-hotel_bookpeople', 'B-hotel_name', 'I-hotel_name', 'B-hotel_stars', 'I-hotel_stars', \
          'B-hotel_type', 'I-hotel_type', 'B-hotel_area', 'B-hotel_pricerange', 'B-restaurant_name', 'I-restaurant_name', 'B-taxi_destination', \
          'I-taxi_destination', 'B-taxi_departure', 'I-taxi_departure', 'B-taxi_arriveby', 'I-taxi_arriveby', 'B-restaurant_booktime', \
          'I-restaurant_booktime', 'B-hotel_parking', 'B-taxi_leaveat', 'I-taxi_leaveat', 'B-restaurant_food', 'I-restaurant_food', \
          'B-attraction_name', 'I-attraction_name', 'B-restaurant_bookpeople', 'B-restaurant_bookday', 'B-restaurant_pricerange', \
          'B-restaurant_area', 'B-train_arriveby', 'I-train_arriveby', 'B-hospital_department', 'I-hospital_department', \
          'B-train_bookpeople', 'B-train_day', 'B-train_leaveat', 'I-train_leaveat', 'B-train_departure', 'I-train_departure', 'B-train_destination', \
          'I-train_destination', 'B-attraction_area', 'B-attraction_type', 'I-attraction_type', 'B-hotel_internet']

# len(y2_coarse_set) =》 32
y2_coarse_set = ['O', 'hotel_bookstay', 'hotel_bookday', 'hotel_bookpeople', 'hotel_name', 'hotel_stars', 'hotel_type', 'hotel_area', \
                 'hotel_pricerange', 'restaurant_name', 'taxi_destination', 'taxi_departure', 'taxi_arriveby', 'restaurant_booktime', \
                 'hotel_parking', 'taxi_leaveat', 'restaurant_food', 'attraction_name', 'restaurant_bookpeople', 'restaurant_bookday', \
                 'restaurant_pricerange', 'restaurant_area', 'train_arriveby', 'hospital_department', 'train_bookpeople', 'train_day', \
                 'train_leaveat', 'train_departure', 'train_destination', 'attraction_area', 'attraction_type', 'hotel_internet']


domain_set = ["Book_Hotel", "Book_Restaurant", "Book_Train", "Find_Attraction", "Find_Hotel", "Find_Restaurant", "Find_Taxi", "Find_Train"]

domain2slot = {'Book_Hotel': ['taxi_arriveby', 'restaurant_booktime', 'taxi_leaveat', 'hotel_parking', 'hotel_bookday', 'hotel_pricerange', 'attraction_name', 'hotel_name', 'hotel_type', 'hotel_stars', 'taxi_destination', 'hotel_area', 'hotel_bookpeople', 'restaurant_food', 'taxi_departure', 'restaurant_name', 'hotel_bookstay'], \
               'Book_Restaurant': ['restaurant_booktime', 'restaurant_area', 'restaurant_bookday', 'taxi_arriveby', 'taxi_leaveat', 'train_arriveby', 'hotel_name', 'taxi_destination', 'restaurant_food', 'hospital_department', 'restaurant_pricerange', 'taxi_departure', 'restaurant_name', 'restaurant_bookpeople'], \
               'Book_Train': ['train_arriveby', 'attraction_name', 'hotel_name', 'train_bookpeople', 'train_destination', 'train_departure', 'restaurant_food', 'train_day', 'train_leaveat', 'restaurant_name'], \
               'Find_Attraction': ['taxi_arriveby', 'taxi_leaveat', 'attraction_name', 'hotel_name', 'taxi_destination', 'restaurant_food', 'restaurant_name', 'hospital_department', 'taxi_departure', 'attraction_area', 'attraction_type'], \
               'Find_Hotel': ['taxi_arriveby', 'hotel_parking', 'hotel_pricerange', 'attraction_name', 'hotel_name', 'hotel_type', 'hotel_stars', 'taxi_destination', 'hotel_area', 'restaurant_food', 'hospital_department', 'taxi_departure', 'restaurant_name', 'hotel_internet'], \
               'Find_Restaurant': ['restaurant_area', 'restaurant_booktime', 'restaurant_bookday', 'attraction_name', 'hotel_name', 'taxi_destination', 'restaurant_food', 'hospital_department', 'restaurant_pricerange', 'taxi_departure', 'restaurant_name', 'restaurant_bookpeople'], \
               'Find_Taxi': ['taxi_arriveby', 'taxi_leaveat', 'restaurant_booktime', 'attraction_name', 'hotel_name', 'taxi_destination', 'hospital_department', 'taxi_departure'], \
               'Find_Train': ['restaurant_booktime', 'taxi_leaveat', 'train_arriveby', 'attraction_name', 'hotel_name', 'train_destination', 'train_bookpeople', 'taxi_destination', 'train_departure', 'restaurant_food', 'train_day', 'taxi_departure', 'train_leaveat', 'restaurant_name']
               }

coarse2id = {'O': 0, 'hotel_bookstay': 1, 'hotel_bookday': 2, 'hotel_bookpeople': 3, 'hotel_name': 4, 'hotel_stars': 5, 'hotel_type': 6, 'hotel_area': 7, 'hotel_pricerange': 8, 'restaurant_name': 9, 'taxi_destination': 10, 'taxi_departure': 11, 'taxi_arriveby': 12, 'restaurant_booktime': 13, 'hotel_parking': 14, 'taxi_leaveat': 15, 'restaurant_food': 16, 'attraction_name': 17, 'restaurant_bookpeople': 18, 'restaurant_bookday': 19, 'restaurant_pricerange': 20, 'restaurant_area': 21, 'train_arriveby': 22, 'hospital_department': 23, 'train_bookpeople': 24, 'train_day': 25, 'train_leaveat': 26, 'train_departure': 27, 'train_destination': 28, 'attraction_area': 29, 'attraction_type': 30, 'hotel_internet': 31}

fine2id = {'O': 0, 'B-hotel_bookstay': 1, 'B-hotel_bookday': 2, 'B-hotel_bookpeople': 3, 'B-hotel_name': 4, 'I-hotel_name': 5, 'B-hotel_stars': 6, 'I-hotel_stars': 7, 'B-hotel_type': 8, 'I-hotel_type': 9, 'B-hotel_area': 10, 'B-hotel_pricerange': 11, 'B-restaurant_name': 12, 'I-restaurant_name': 13, 'B-taxi_destination': 14, 'I-taxi_destination': 15, 'B-taxi_departure': 16, 'I-taxi_departure': 17, 'B-taxi_arriveby': 18, 'I-taxi_arriveby': 19, 'B-restaurant_booktime': 20, 'I-restaurant_booktime': 21, 'B-hotel_parking': 22, 'B-taxi_leaveat': 23, 'I-taxi_leaveat': 24, 'B-restaurant_food': 25, 'I-restaurant_food': 26, 'B-attraction_name': 27, 'I-attraction_name': 28, 'B-restaurant_bookpeople': 29, 'B-restaurant_bookday': 30, 'B-restaurant_pricerange': 31, 'B-restaurant_area': 32, 'B-train_arriveby': 33, 'I-train_arriveby': 34, 'B-hospital_department': 35, 'I-hospital_department': 36, 'B-train_bookpeople': 37, 'B-train_day': 38, 'B-train_leaveat': 39, 'I-train_leaveat': 40, 'B-train_departure': 41, 'I-train_departure': 42, 'B-train_destination': 43, 'I-train_destination': 44, 'B-attraction_area': 45, 'B-attraction_type': 46, 'I-attraction_type': 47, 'B-hotel_internet': 48}



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
                if slot not in BIO_with_slot_dict.keys():  # some slots may not be in utterance: just add list of "O"s
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

    domain_set = ["Book_Hotel", "Book_Restaurant", "Book_Train", "Find_Attraction", "Find_Hotel", "Find_Restaurant",
                  "Find_Taxi", "Find_Train"]

    data = {"Book_Hotel": {}, "Book_Restaurant": {}, "Book_Train": {}, "Find_Attraction": {}, "Find_Hotel": {},
            "Find_Restaurant": {}, "Find_Taxi": {}, "Find_Train": {}}


    max_length = {"Book_Hotel": 0, "Book_Restaurant": 0, "Book_Train": 0, "Find_Attraction": 0, "Find_Hotel": 0,
                  "Find_Restaurant": 0, "Find_Taxi": 0, "Find_Train": 0}

    data_size = {"Book_Hotel": 0, "Book_Restaurant": 0, "Book_Train": 0, "Find_Attraction": 0, "Find_Hotel": 0,
                  "Find_Restaurant": 0, "Find_Taxi": 0, "Find_Train": 0}

    # load data
    data['Book_Hotel'], max_length['Book_Hotel'], data_size['Book_Hotel'] = read_file(f"{data_path}/Book_Hotel/Book_Hotel.txt",
                                                                   domain="Book_Hotel", use_dep=use_dep, use_pos=use_pos)
    data['Book_Restaurant'], max_length['Book_Restaurant'], data_size['Book_Restaurant'] = read_file(f"{data_path}/Book_Restaurant/Book_Restaurant.txt",
                                                                     domain="Book_Restaurant", use_dep=use_dep, use_pos=use_pos)
    data['Book_Train'], max_length['Book_Train'], data_size['Book_Train'] = read_file(f"{data_path}/Book_Train/Book_Train.txt",
                                                             domain="Book_Train", use_dep=use_dep, use_pos=use_pos)
    data['Find_Attraction'], max_length['Find_Attraction'], data_size['Find_Attraction'] = read_file(f"{data_path}/Find_Attraction/Find_Attraction.txt", domain="Find_Attraction", use_dep=use_dep, use_pos=use_pos)
    data['Find_Hotel'], max_length['Find_Hotel'], data_size['Find_Hotel'] = read_file(f"{data_path}/Find_Hotel/Find_Hotel.txt", domain="Find_Hotel",use_dep=use_dep, use_pos=use_pos)
    data['Find_Restaurant'], max_length['Find_Restaurant'], data_size['Find_Restaurant'] = read_file(
        f"{data_path}/Find_Restaurant/Find_Restaurant.txt", domain="Find_Restaurant", use_dep=use_dep, use_pos=use_pos)
    data['Find_Taxi'], max_length['Find_Taxi'], data_size['Find_Taxi'] = read_file(
        f"{data_path}/Find_Taxi/Find_Taxi.txt", domain="Find_Taxi", use_dep=use_dep, use_pos=use_pos)
    data['Find_Train'], max_length['Find_Train'], data_size['Find_Train'] = read_file(
        f"{data_path}/Find_Train/Find_Train.txt", domain="Find_Train", use_dep=use_dep, use_pos=use_pos)

    # # unseen
    # data['Book_Hotel_unseen'], max_length['Book_Hotel_unseen'], data_size['Book_Hotel_unseen'] = read_file(
    #     f"{data_path}/Book_Hotel/unseen_slots.txt",
    #     domain="Book_Hotel", use_dep=use_dep, use_pos=use_pos)
    # data['Book_Restaurant_unseen'], max_length['Book_Restaurant_unseen'], data_size['Book_Restaurant_unseen'] = read_file(
    #     f"{data_path}/Book_Restaurant/unseen_slots.txt",
    #     domain="Book_Restaurant", use_dep=use_dep, use_pos=use_pos)
    # data['Book_Train_unseen'], max_length['Book_Train_unseen'], data_size['Book_Train_unseen'] = read_file(
    #     f"{data_path}/Book_Train/unseen_slots.txt",
    #     domain="Book_Train", use_dep=use_dep, use_pos=use_pos)
    # data['Find_Attraction_unseen'], max_length['Find_Attraction_unseen'], data_size['Find_Attraction_unseen'] = read_file(
    #     f"{data_path}/Find_Attraction/unseen_slots.txt", domain="Find_Attraction", use_dep=use_dep, use_pos=use_pos)
    # data['Find_Hotel_unseen'], max_length['Find_Hotel_unseen'], data_size['Find_Hotel_unseen'] = read_file(f"{data_path}/Find_Hotel/unseen_slots.txt",
    #                                                                             domain="Find_Hotel", use_dep=use_dep,
    #                                                                             use_pos=use_pos)
    # data['Find_Restaurant_unseen'], max_length['Find_Restaurant_unseen'], data_size['Find_Restaurant_unseen'] = read_file(
    #     f"{data_path}/Find_Restaurant/unseen_slots.txt", domain="Find_Restaurant", use_dep=use_dep,
    #     use_pos=use_pos)
    # data['Find_Taxi_unseen'], max_length['Find_Taxi_unseen'], data_size['Find_Taxi_unseen'] = read_file(
    #     f"{data_path}/Find_Taxi/unseen_slots.txt", domain="Find_Taxi", use_dep=use_dep,
    #     use_pos=use_pos)

    # # seen
    # data['Book_Hotel_seen'], max_length['Book_Hotel_seen'], data_size['Book_Hotel_seen'] = read_file(
    #     f"{data_path}/Book_Hotel/seen_slots.txt",
    #     domain="Book_Hotel", use_dep=use_dep, use_pos=use_pos)
    # data['Book_Restaurant_seen'], max_length['Book_Restaurant_seen'], data_size['Book_Restaurant_seen'] = read_file(
    #     f"{data_path}/Book_Restaurant/seen_slots.txt",
    #     domain="Book_Restaurant", use_dep=use_dep, use_pos=use_pos)
    # data['Book_Train_seen'], max_length['Book_Train_seen'], data_size['Book_Train_seen'] = read_file(
    #     f"{data_path}/Book_Train/seen_slots.txt",
    #     domain="Book_Train", use_dep=use_dep, use_pos=use_pos)
    # data['Find_Attraction_seen'], max_length['Find_Attraction_seen'], data_size['Find_Attraction_seen'] = read_file(
    #     f"{data_path}/Find_Attraction/seen_slots.txt", domain="Find_Attraction", use_dep=use_dep, use_pos=use_pos)
    # data['Find_Hotel_seen'], max_length['Find_Hotel_seen'], data_size['Find_Hotel_seen'] = read_file(f"{data_path}/Find_Hotel/seen_slots.txt",
    #                                                                             domain="Find_Hotel", use_dep=use_dep,
    #                                                                             use_pos=use_pos)
    # data['Find_Restaurant_seen'], max_length['Find_Restaurant_seen'], data_size['Find_Restaurant_seen'] = read_file(
    #     f"{data_path}/Find_Restaurant/seen_slots.txt", domain="Find_Restaurant", use_dep=use_dep,
    #     use_pos=use_pos)
    # data['Find_Taxi_seen'], max_length['Find_Taxi_seen'], data_size['Find_Taxi_seen'] = read_file(
    #     f"{data_path}/Find_Taxi/seen_slots.txt", domain="Find_Taxi", use_dep=use_dep,
    #     use_pos=use_pos)

    # data_atp, max_length['Book_Hotel'] = read_file(f"{data_path}/Book_Hotel/Book_Hotel.txt", domain="Book_Hotel")
    # data_br, max_length['Book_Restaurant'] = read_file(f"{data_path}/Book_Restaurant/Book_Restaurant.txt", domain="Book_Restaurant")
    # data_gw, max_length['Book_Train'] = read_file(f"{data_path}/Book_Train/Book_Train.txt", domain="Book_Train")
    # data_pm, max_length['Find_Attraction'] = read_file(f"{data_path}/Find_Attraction/Find_Attraction.txt", domain="Find_Attraction")
    # data_rb, max_length['Find_Hotel'] = read_file(f"{data_path}/Find_Hotel/Find_Hotel.txt", domain="Find_Hotel")
    # data_scw, max_length['Find_Restaurant'] = read_file(f"{data_path}/Find_Restaurant/Find_Restaurant.txt", domain="Find_Restaurant")
    # data_sse, max_length['Find_Taxi'] = read_file(f"{data_path}/Find_Taxi/Find_Taxi.txt", domain="Find_Taxi")

    # data["Book_Hotel"] =  data_binarize(data_atp)
    # data["Book_Restaurant"] =  data_binarize(data_br)
    # data["Book_Train"] =  data_binarize(data_gw)
    # data["Find_Attraction"] =  data_binarize(data_pm)
    # data["Find_Hotel"] =  data_binarize(data_rb)
    # data["Find_Restaurant"] =  data_binarize(data_scw)
    # data["Find_Taxi"] =  data_binarize(data_sse)

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
