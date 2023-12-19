from src.datareader import datareader, PAD_INDEX, domain2slot, y2_set
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import math

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class Dataset(data.Dataset):
    def __init__(self, domain, label, X, y, X_dep, X_pos, coarse_label, fine_label,  max_len, tokenizer):
        self.domain = domain
        self.label = label
        self.X = X
        self.y = y
        self.X_dep = X_dep
        self.X_pos = X_pos
        self.coarse_label = coarse_label
        self.fine_label = fine_label
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        slot = self.label[index].split()
        utter = self.X[index].split()


        if self.X_dep != [] and self.X_pos != []:
            utter_dep = self.X_dep[index].split()
            utter_pos = self.X_pos[index].split()
            encoded = self.tokenizer(slot, utter_dep, utter_pos, utter, is_split_into_words=True)
            subword_ids = encoded.word_ids()
            labels = self.y[index]
            gold_coarse_labels = self.coarse_label[index]
            gold_fine_labels = self.fine_label[index]
            new_labels = []
            new_gold_coarse_label = []
            new_gold_fine_label = []
            none_counter = 0
            for i, word_idx in enumerate(subword_ids):
                if none_counter < 4 or word_idx is None:
                    new_labels.append(0)
                    if word_idx is None:

                        none_counter += 1
                elif none_counter == 4:
                    new_labels.append(labels[word_idx])
        elif self.X_pos != []:
            pass

        elif self.X_dep != []:
            # 'input_ids': encoded['input_ids'],
            # 'attention_mask': encoded['attention_mask'],
            # 'token_type_ids': encoded['token_type_ids'],
            encoded = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}

            utter_dep = self.X_dep[index].split()
            encoded1 = self.tokenizer(slot, utter_dep, is_split_into_words=True)
            encoded2 = self.tokenizer(utter, is_split_into_words=True)

            subword_ids_1 = encoded1.word_ids()
            subword_ids_2 = encoded2.word_ids()
            """
            self.tokenizer.tokenize(' '.join(
                ['[CLS]', 'object', '_', 'name', '[SEP]', 'compound', 'root', 'prep', 'compound', 'pobj', 'nummod',
                 'appos', '[SEP]', 'rate', 'adventures', 'in', 'stationery', 'saga', '5', 'stars', '[SEP]']))
            ['[CLS]', 'object', '_', 'name', '[SEP]', 'compound', 'root', 'prep', 'compound', 'po', '##b', '##j', 'nu',
             '##mm', '##od', 'app', '##os', '[SEP]', 'rate', 'adventures', 'in', 'station', '##ery', 'saga', '5',
             'stars', '[SEP]']
            len(['[CLS]', 'object', '_', 'name', '[SEP]', 'compound', 'root', 'prep', 'compound', 'po', '##b', '##j',
                 'nu', '##mm', '##od', 'app', '##os', '[SEP]', 'rate', 'adventures', 'in', 'station', '##ery', 'saga',
                 '5', 'stars', '[SEP]'])
            27
            """
            # [None, 0, None, 0, 0, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, None, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, None]
            subword_ids = subword_ids_1 + subword_ids_2[1:]

            """
            e2=self.tokenizer(utter, is_split_into_words=True)
            {'input_ids': [101, 2054, 2024, 1996, 4633, 3785, 1999, 6986, 23692, 6200, 2148, 3088, 102],
             'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
             e2.word_ids()
             [None, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, None]
            """
            # len('[CLS] object _ name [SEP] compound root prep compound pobj nummod appos [SEP] rate adventures in stationery saga 5 stars [SEP]'.split())
            # 21
            encoded['input_ids'] = encoded1['input_ids'] + encoded2['input_ids'][1:]
            encoded['token_type_ids'] = encoded1['token_type_ids'] + encoded2['token_type_ids'][1:]
            encoded['attention_mask'] = encoded1['attention_mask'] + encoded2['attention_mask'][1:]

            labels = self.y[index]
            gold_coarse_labels = self.coarse_label[index]
            gold_fine_labels = self.fine_label[index]
            new_labels = []
            new_gold_coarse_labels = []
            new_gold_fine_labels = []
            none_counter = 0
            for i, word_idx in enumerate(subword_ids):
                if none_counter < 3 or word_idx is None:
                    new_labels.append(0)
                    new_gold_coarse_labels.append(0)
                    new_gold_fine_labels.append(0)
                    if word_idx is None:
                        none_counter += 1

                elif none_counter == 3:
                    new_labels.append(labels[word_idx])
                    new_gold_coarse_labels.append(gold_coarse_labels[word_idx])
                    new_gold_fine_labels.append(gold_fine_labels[word_idx])

        elif self.X_pos !=[]:
            utter_pos = self.X_pos[index].split()
            encoded = self.tokenizer(slot, utter_pos, utter, is_split_into_words=True)
            subword_ids = encoded.word_ids()
            labels = self.y[index]
            new_labels = []

            none_counter = 0
            for i, word_idx in enumerate(subword_ids):
                if none_counter < 3 or word_idx is None:
                    new_labels.append(0)
                    if word_idx is None:
                        none_counter += 1

                elif none_counter == 3:
                    new_labels.append(labels[word_idx])
        else:
            encoded = self.tokenizer(slot, utter, is_split_into_words=True)
            ### BERT tokenizes sequence into subword tokens --> BIO labels should be modified to match with them
            subword_ids = encoded.word_ids()
            labels = self.y[index]
            gold_coarse_labels = self.coarse_label[index]
            gold_fine_labels = self.fine_label[index]
            new_labels = []
            new_gold_coarse_labels = []
            new_gold_fine_labels = []

            none_counter = 0
            for i, word_idx in enumerate(subword_ids):
                if none_counter < 2 or word_idx is None:
                    new_labels.append(0)
                    new_gold_coarse_labels.append(0)
                    new_gold_fine_labels.append(0)
                    if word_idx is None:
                        none_counter += 1

                elif none_counter == 2:
                    new_labels.append(labels[word_idx])
                    new_gold_coarse_labels.append(gold_coarse_labels[word_idx])
                    new_gold_fine_labels.append(gold_fine_labels[word_idx])

        # y = []
        # for i, word in enumerate(self.X[index].split()):
        #     tokenized = self.tokenizer.tokenize(word)
        #     num_subwords = len(tokenized)

        #     y.extend([self.y[index][i]] * num_subwords)
        assert len(encoded['input_ids']) == len(encoded['attention_mask']) == len(encoded['token_type_ids']) == len(new_labels)
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded['token_type_ids'],
            'labels': new_labels,
            'gold_coarse_labels': new_gold_coarse_labels,
            'gold_fine_labels' : new_gold_fine_labels

        }
    
    def __len__(self):
        return len(self.X)


def pad_tensor(features, max_seq_len):
# def pad_tensor(dim0, dim1, tensor, front_pad=None):
    """
    features: list of lists, each element equals to output of hf tokenizer
    """
    padded_features = []
    for f in features:
        original_value_len = len(f)
        for _ in range(original_value_len, max_seq_len):
            f.append(PAD_INDEX)
        
        padded_features.append(f)

    return torch.tensor(padded_features, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(padded_features, dtype=torch.long)


def collate_fn(features):
    """
    Collate function for SLU Model
    pad at right side
    """
    padded_features = {}
    batch_size = len(features)
    max_seq_len = 0

    _batch = {}
    for f in features:
        for k, v in f.items():
            try:
                padded_features[k].append(v)
            except:
                padded_features[k] = [v]
        
            feature_len = len(v)
            if feature_len > max_seq_len:
                max_seq_len = feature_len

    for k, v in padded_features.items():
        v = pad_tensor(v, max_seq_len)
        padded_features[k] = v


    return padded_features


def get_dataloader(tgt_domain, batch_size, n_samples, data_path, tokenizer, use_dep, use_pos):
    # data_dict = {"domain": domain_list, "label": label_list, "utter": utter_list, "y": y_list,
    #              "utter_dep": utter_dep_list, "utter_pos": utter_pos_list}
    data_size = {}
    # data_size = {"AddToPlaylist": 0, "BookRestaurant": 0, "GetWeather": 0, "PlayMusic": 0, "RateBook": 0,
    #              "SearchCreativeWork": 0, "SearchScreeningEvent": 0}
    dm_data_size = {'AddToPlaylist': 2042, 'BookRestaurant': 2073, 'GetWeather': 2100, 'PlayMusic': 2100, 'RateBook': 2056,
     'SearchCreativeWork': 2054, 'SearchScreeningEvent': 2059, 'AddToPlaylist_unseen': 1054,
     'BookRestaurant_unseen': 1514, 'GetWeather_unseen': 962, 'PlayMusic_unseen': 1194, 'RateBook_unseen': 1542,
     'SearchCreativeWork_unseen': 0, 'SearchScreeningEvent_unseen': 1379, 'AddToPlaylist_seen': 477,
     'BookRestaurant_seen': 40, 'GetWeather_seen': 613, 'PlayMusic_seen': 381, 'RateBook_seen': 0,
     'SearchCreativeWork_seen': 1540, 'SearchScreeningEvent_seen': 165}

    domains = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

    all_data, max_length, data_size = datareader(data_path, use_dep, use_pos)
    train_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": [], "gold_coarse_label": [], "gold_fine_label": []}

    # if use_dep and use_pos:
    #     train_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": []}
    # elif use_dep:
    #     train_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": []}
    # elif use_pos:
    #     train_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_pos": []}
    # else:
    #     train_data = {"domain": [], "label": [], "utter": [], "y": []}

    # print("all_data.keys(): ", all_data.keys())

    for dm_name, dm_data in all_data.items():
        # if dm_name != tgt_domain and 'unseen' not in dm_name and 'seen' not in dm_name:
        if dm_name != tgt_domain:
            # print('dm_name: {}'.format(dm_name))
            train_data["domain"].extend(dm_data["domain"])
            train_data["label"].extend(dm_data["label"])
            train_data["utter"].extend(dm_data["utter"])
            train_data["y"].extend(dm_data["y"])
            train_data["gold_coarse_label"].extend(dm_data["gold_coarse_label"])
            train_data["gold_fine_label"].extend(dm_data["gold_fine_label"])
            if use_dep and use_pos:
                train_data["utter_dep"].extend(dm_data["utter_dep"])
                train_data["utter_pos"].extend(dm_data["utter_pos"])
            elif use_dep:
                train_data["utter_dep"].extend(dm_data["utter_dep"])
            elif use_pos:
                train_data["utter_pos"].extend(dm_data["utter_pos"])
    val_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": [], "gold_coarse_label": [], "gold_fine_label": []}
    test_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": [], "gold_coarse_label": [], "gold_fine_label": []}
    unseen_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": [], "gold_coarse_label": [], "gold_fine_label": []}
    seen_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": [], "gold_coarse_label": [], "gold_fine_label": []}

    # if use_dep and use_pos:
    #     val_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": []}
    #     test_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": [], "utter_pos": []}
    # elif use_dep:
    #     val_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": []}
    #     test_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_dep": []}
    # elif use_pos:
    #     val_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_pos": []}
    #     test_data = {"domain": [], "label": [], "utter": [], "y": [], "utter_pos": []}
    # else:
    #     val_data = {"domain": [], "label": [], "utter": [], "y": []}
    #     test_data = {"domain": [], "label": [], "utter": [], "y": []}

    # num_tgt_slots = len(domain2slot[tgt_domain])
    # val_split = 500 * num_tgt_slots  # validation: 500 utterances
    # train_split = n_samples * num_tgt_slots
    tgt_data_size = data_size[tgt_domain]
    # print("tgt_data_size: ", tgt_data_size)

    valid_split_ratio = 500/tgt_data_size
    # print("valid_split_ratio: ", valid_split_ratio)

    train_split_ratio = n_samples/tgt_data_size
    # print("train_split_ratio: ", train_split_ratio)

    val_split = math.ceil(valid_split_ratio * len(all_data[tgt_domain]["utter"]))
    # print("len(all_data[tgt_domain]['utter'])", len(all_data[tgt_domain]["utter"]))
    print("val_split: ", val_split)
    train_split = math.ceil(train_split_ratio * len(all_data[tgt_domain]["utter"]))
    print("train_split: ", train_split)

    # unseen and seen data generation
    tgt_domain_to_all_seen_slots = []
    for src_dm in domains:
        if src_dm != tgt_domain:
            tgt_domain_to_all_seen_slots.extend(domain2slot[src_dm])
    # remove repeated slot type
    tgt_domain_to_all_seen_slots = list(set(tgt_domain_to_all_seen_slots))

    unseen_slot_cnt, seen_slot_cnt = 0, 0
    tgt_data = all_data[tgt_domain]
    for dm, label, utter, y, utter_dep, utter_pos, coarse_label, fine_label in zip(tgt_data["domain"][val_split:], tgt_data["label"][val_split:], \
                                                                                   tgt_data["utter"][val_split:], tgt_data["y"][val_split:],tgt_data["utter_dep"][val_split:],\
                                                                                   tgt_data["utter_pos"][val_split:],tgt_data["gold_coarse_label"][val_split:], tgt_data["gold_fine_label"][val_split:]):
        if label in tgt_domain_to_all_seen_slots:
            seen_slot_cnt += 1
            seen_data["domain"].extend(dm)
            seen_data["label"].extend(label)
            seen_data["utter"].extend(utter)
            seen_data["y"].extend(y)
            if use_dep and use_pos:
                seen_data["utter_dep"].extend(utter_dep)
                seen_data["utter_pos"].extend(utter_pos)
            elif use_dep:
                seen_data["utter_dep"].extend(utter_dep)
            elif use_pos:
                seen_data["utter_pos"].extend(utter_pos)
            seen_data["gold_coarse_label"].extend(coarse_label)
            seen_data["gold_fine_label"].extend(fine_label)
        else:
            unseen_slot_cnt += 1
            unseen_data["domain"].extend(dm)
            unseen_data["label"].extend(label)
            unseen_data["utter"].extend(utter)
            unseen_data["y"].extend(y)
            if use_dep and use_pos:
                unseen_data["utter_dep"].extend(utter_dep)
                unseen_data["utter_pos"].extend(utter_pos)
            elif use_dep:
                unseen_data["utter_dep"].extend(utter_dep)
            elif use_pos:
                unseen_data["utter_pos"].extend(utter_pos)
            unseen_data["gold_coarse_label"].extend(coarse_label)
            unseen_data["gold_fine_label"].extend(fine_label)
    print("Target Domain: %s; Seen samples: %d; Unseen samples: %d; Total samples: %d" % (
    tgt_domain, seen_slot_cnt, unseen_slot_cnt, unseen_slot_cnt + seen_slot_cnt))

    if n_samples == 0:
        if use_dep and use_pos:
            # first 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][:val_split]
            val_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][:val_split]
            val_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][:val_split]

            # the rest as test set
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][val_split:]
            test_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]

        elif use_dep:
            # first 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][:val_split]
            val_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][:val_split]

            # the rest as test set
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]

        elif use_pos:
            val_data["domain"] = all_data[tgt_domain]["domain"][:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][:val_split]
            val_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][:val_split]

            # the rest as test set
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]

        else:
            # first 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][:val_split]

            # the rest as test set
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]

    else:
        if use_dep and use_pos:
            # first n samples as train set
            train_data["domain"].extend(all_data[tgt_domain]["domain"][:train_split])
            train_data["label"].extend(all_data[tgt_domain]["label"][:train_split])
            train_data["utter"].extend(all_data[tgt_domain]["utter"][:train_split])
            train_data["y"].extend(all_data[tgt_domain]["y"][:train_split])
            train_data["utter_dep"].extend(all_data[tgt_domain]["utter_dep"][:train_split])
            train_data["utter_pos"].extend(all_data[tgt_domain]["utter_pos"][:train_split])
            train_data["gold_coarse_label"].extend(all_data[tgt_domain]["gold_coarse_label"][:train_split])
            train_data["gold_fine_label"].extend(all_data[tgt_domain]["gold_fine_label"][:train_split])

            # from n to 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][train_split:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][train_split:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][train_split:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][train_split:val_split]
            val_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][train_split:val_split]
            val_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][train_split:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][train_split:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][train_split:val_split]

            # the rest as test set (same as zero-shot)
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][val_split:]
            test_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]

        elif use_dep:
            # first n samples as train set
            train_data["domain"].extend(all_data[tgt_domain]["domain"][:train_split])
            train_data["label"].extend(all_data[tgt_domain]["label"][:train_split])
            train_data["utter"].extend(all_data[tgt_domain]["utter"][:train_split])
            train_data["y"].extend(all_data[tgt_domain]["y"][:train_split])
            train_data["utter_dep"].extend(all_data[tgt_domain]["utter_dep"][:train_split])
            train_data["gold_coarse_label"].extend(all_data[tgt_domain]["gold_coarse_label"][:train_split])
            train_data["gold_fine_label"].extend(all_data[tgt_domain]["gold_fine_label"][:train_split])

            # from n to 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][train_split:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][train_split:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][train_split:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][train_split:val_split]
            val_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][train_split:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][train_split:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][train_split:val_split]

            # the rest as test set (same as zero-shot)
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["utter_dep"] = all_data[tgt_domain]["utter_dep"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]


        elif use_pos:
            # first n samples as train set
            train_data["domain"].extend(all_data[tgt_domain]["domain"][:train_split])
            train_data["label"].extend(all_data[tgt_domain]["label"][:train_split])
            train_data["utter"].extend(all_data[tgt_domain]["utter"][:train_split])
            train_data["y"].extend(all_data[tgt_domain]["y"][:train_split])
            train_data["utter_pos"].extend(all_data[tgt_domain]["utter_pos"][:train_split])
            train_data["gold_coarse_label"].extend(all_data[tgt_domain]["gold_coarse_label"][:train_split])
            train_data["gold_fine_label"].extend(all_data[tgt_domain]["gold_fine_label"][:train_split])

            # from n to 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][train_split:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][train_split:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][train_split:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][train_split:val_split]
            val_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][train_split:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][train_split:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][train_split:val_split]

            # the rest as test set (same as zero-shot)
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["utter_pos"] = all_data[tgt_domain]["utter_pos"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]


        else:
            # first n samples as train set
            train_data["domain"].extend(all_data[tgt_domain]["domain"][:train_split])
            train_data["label"].extend(all_data[tgt_domain]["label"][:train_split])
            train_data["utter"].extend(all_data[tgt_domain]["utter"][:train_split])
            train_data["y"].extend(all_data[tgt_domain]["y"][:train_split])
            train_data["gold_coarse_label"].extend(all_data[tgt_domain]["gold_coarse_label"][:train_split])
            train_data["gold_fine_label"].extend(all_data[tgt_domain]["gold_fine_label"][:train_split])

            # from n to 500 samples as validation set
            val_data["domain"] = all_data[tgt_domain]["domain"][train_split:val_split]
            val_data["label"] = all_data[tgt_domain]["label"][train_split:val_split]
            val_data["utter"] = all_data[tgt_domain]["utter"][train_split:val_split]
            val_data["y"] = all_data[tgt_domain]["y"][train_split:val_split]
            val_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][train_split:val_split]
            val_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][train_split:val_split]

            # the rest as test set (same as zero-shot)
            test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
            test_data["label"] = all_data[tgt_domain]["label"][val_split:]
            test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
            test_data["y"] = all_data[tgt_domain]["y"][val_split:]
            test_data["gold_coarse_label"] = all_data[tgt_domain]["gold_coarse_label"][val_split:]
            test_data["gold_fine_label"] = all_data[tgt_domain]["gold_fine_label"][val_split:]

    # if use_dep and use_pos:
    #     dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"], train_data["utter_dep"], train_data["utter_pos"],
    #                          max_length, tokenizer)
    #     dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"], val_data["utter_dep"], val_data["utter_pos"], max_length,
    #                           tokenizer)
    #     dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"], test_data["utter_dep"], test_data["utter_pos"], max_length,
    #                            tokenizer)
    # elif use_dep:
    #     dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"], train_data["utter_dep"],
    #                          max_length, tokenizer)
    #     dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"], val_data["utter_dep"], max_length,
    #                           tokenizer)
    #     dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"], test_data["utter_dep"], max_length,
    #                            tokenizer)
    #
    # elif use_pos:
    #     dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"], train_data["utter_pos"],
    #                          max_length, tokenizer)
    #     dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"], max_length, val_data["utter_pos"],
    #                           tokenizer)
    #     dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"], max_length, test_data["utter_pos"],
    #                            tokenizer)
    #
    # else:
    #     dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"], max_length, tokenizer)
    #     dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"], max_length, tokenizer)
    #     dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"], max_length, tokenizer)

    dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"],
                         train_data["utter_dep"], train_data["utter_pos"], train_data["gold_coarse_label"], train_data["gold_fine_label"],
                         max_length, tokenizer)
    dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"],
                          val_data["utter_dep"], val_data["utter_pos"], val_data["gold_coarse_label"], val_data["gold_fine_label"],
                          max_length, tokenizer)
    dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"],
                           test_data["utter_dep"], test_data["utter_pos"], test_data["gold_coarse_label"], test_data["gold_fine_label"],
                           max_length, tokenizer)

    dataset_unseen = Dataset(unseen_data["domain"], unseen_data["label"], unseen_data["utter"], unseen_data["y"],
                         unseen_data["utter_dep"], unseen_data["utter_pos"], unseen_data["gold_coarse_label"], unseen_data["gold_fine_label"],
                         max_length, tokenizer)

    dataset_seen = Dataset(seen_data["domain"], seen_data["label"], seen_data["utter"], seen_data["y"],
                         seen_data["utter_dep"], seen_data["utter_pos"], seen_data["gold_coarse_label"], seen_data["gold_fine_label"],
                         max_length, tokenizer)


    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # dataloader_test = DataLoader(dataset=dataset_test, batch_size=len(domain2slot[tgt_domain]), shuffle=False,
    #                              collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn)
    dataloader_unseen = DataLoader(dataset=dataset_unseen, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_seen = DataLoader(dataset=dataset_seen, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test, dataloader_unseen, dataloader_seen