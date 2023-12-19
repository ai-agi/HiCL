
y2_set = ['O', 'B-hotel_bookstay', 'B-hotel_bookday', 'B-hotel_bookpeople', 'B-hotel_name', 'I-hotel_name', 'B-hotel_stars', 'I-hotel_stars', \
          'B-hotel_type', 'I-hotel_type', 'B-hotel_area', 'B-hotel_pricerange', 'B-restaurant_name', 'I-restaurant_name', 'B-taxi_destination', \
          'I-taxi_destination', 'B-taxi_departure', 'I-taxi_departure', 'B-taxi_arriveby', 'I-taxi_arriveby', 'B-restaurant_booktime', \
          'I-restaurant_booktime', 'B-hotel_parking', 'B-taxi_leaveat', 'I-taxi_leaveat', 'B-restaurant_food', 'I-restaurant_food', \
          'B-attraction_name', 'I-attraction_name', 'B-restaurant_bookpeople', 'B-restaurant_bookday', 'B-restaurant_pricerange', \
          'B-restaurant_area', 'B-train_arriveby', 'I-train_arriveby', 'B-hospital_department', 'I-hospital_department', \
          'B-train_bookpeople', 'B-train_day', 'B-train_leaveat', 'I-train_leaveat', 'B-train_departure', 'I-train_departure', 'B-train_destination', \
          'I-train_destination', 'B-attraction_area', 'B-attraction_type', 'I-attraction_type', 'B-hotel_internet']


y2_coarse_set = ['O', 'hotel_bookstay', 'hotel_bookday', 'hotel_bookpeople', 'hotel_name', 'hotel_stars', 'hotel_type', 'hotel_area', \
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
               'Find_Train': ['restaurant_booktime', 'taxi_leaveat', 'train_arriveby', 'attraction_name', 'hotel_name', 'train_destination', 'train_bookpeople', 'taxi_destination', 'train_departure', 'restaurant_food', 'train_day', 'taxi_departure', 'train_leaveat', 'restaurant_name']}


coarse2id = {'O': 0, 'hotel_bookstay': 1, 'hotel_bookday': 2, 'hotel_bookpeople': 3, 'hotel_name': 4, 'hotel_stars': 5, 'hotel_type': 6, 'hotel_area': 7, 'hotel_pricerange': 8, 'restaurant_name': 9, 'taxi_destination': 10, 'taxi_departure': 11, 'taxi_arriveby': 12, 'restaurant_booktime': 13, 'hotel_parking': 14, 'taxi_leaveat': 15, 'restaurant_food': 16, 'attraction_name': 17, 'restaurant_bookpeople': 18, 'restaurant_bookday': 19, 'restaurant_pricerange': 20, 'restaurant_area': 21, 'train_arriveby': 22, 'hospital_department': 23, 'train_bookpeople': 24, 'train_day': 25, 'train_leaveat': 26, 'train_departure': 27, 'train_destination': 28, 'attraction_area': 29, 'attraction_type': 30, 'hotel_internet': 31}

fine2id = {'O': 0, 'B-hotel_bookstay': 1, 'B-hotel_bookday': 2, 'B-hotel_bookpeople': 3, 'B-hotel_name': 4, 'I-hotel_name': 5, 'B-hotel_stars': 6, 'I-hotel_stars': 7, 'B-hotel_type': 8, 'I-hotel_type': 9, 'B-hotel_area': 10, 'B-hotel_pricerange': 11, 'B-restaurant_name': 12, 'I-restaurant_name': 13, 'B-taxi_destination': 14, 'I-taxi_destination': 15, 'B-taxi_departure': 16, 'I-taxi_departure': 17, 'B-taxi_arriveby': 18, 'I-taxi_arriveby': 19, 'B-restaurant_booktime': 20, 'I-restaurant_booktime': 21, 'B-hotel_parking': 22, 'B-taxi_leaveat': 23, 'I-taxi_leaveat': 24, 'B-restaurant_food': 25, 'I-restaurant_food': 26, 'B-attraction_name': 27, 'I-attraction_name': 28, 'B-restaurant_bookpeople': 29, 'B-restaurant_bookday': 30, 'B-restaurant_pricerange': 31, 'B-restaurant_area': 32, 'B-train_arriveby': 33, 'I-train_arriveby': 34, 'B-hospital_department': 35, 'I-hospital_department': 36, 'B-train_bookpeople': 37, 'B-train_day': 38, 'B-train_leaveat': 39, 'I-train_leaveat': 40, 'B-train_departure': 41, 'I-train_departure': 42, 'B-train_destination': 43, 'I-train_destination': 44, 'B-attraction_area': 45, 'B-attraction_type': 46, 'I-attraction_type': 47, 'B-hotel_internet': 48}




def multiwoz_domain_slot_generate(filepath, domain, y2_set, y2_coarse_set, domain2slot):
    with open(filepath, "r") as f:
        slots = []
        for i, line in enumerate(f):
            line = line.strip()  # query \t BIO-labels
            splits = line.split("\t")  # split query and labels
            utter = splits[0]
            labels = splits[1].split()
            for label in labels:
                if 'B-' in label and label:
                    y2_set.append(label)
                    slots.append(label.replace('B-', ''))
                    y2_coarse_set.append(label.replace('B-', ''))
                elif 'I-' in label:
                    y2_set.append(label)
                    # slots.append(label)

    slots = list(set(slots))
    domain2slot[domain] = slots
    y2_set = list(set(y2_set))
    y2_coarse_set = list(set(y2_coarse_set))

    print("y2_set: {}".format(y2_set))
    print("len(y2_set): {}".format(len(y2_set)))
    return y2_set, y2_coarse_set, domain2slot

if __name__ == "__main__":
    # y2_set = ['O']
    # y2_coarse_set = ['O']
    # domain2slot = {}
    coarse2id = {}
    fine2id = {}
    #
    # domains = ["Book_Hotel", "Book_Restaurant", "Book_Train", "Find_Attraction", "Find_Hotel", "Find_Restaurant", "Find_Taxi", "Find_Train"]
    # for domain in domains:
    #     filepath = "../data/multiwoz/extract/" + domain + "/" + domain + ".txt"
    #     y2_set, y2_coarse_set, domain2slot = multiwoz_domain_slot_generate(filepath, domain, y2_set, y2_coarse_set, domain2slot)


    for i, label in enumerate(_y2_coarse_set):
        coarse2id[label] = i

    for i, label in enumerate(_y2_set):
        fine2id[label] = i

    # print("y2_set: {}".format(_y2_set))
    # print("y2_coarse_set: {}".format(_y2_coarse_set))
    # print("len(y2_set): {}".format(len(_y2_set)))
    # print("len(y2_coarse_set): {}".format(len(y2_coarse_set)))
    #
    # print("len(_y2_set): {}".format(len(_y2_set)))
    # print("len(_y2_coarse_set): {}".format(len(_y2_coarse_set)))
    #
    # print("domain2slot: {}".format(domain2slot))
    print("coarse2id: {}".format(coarse2id))
    print("fine2id: {}".format(fine2id))

