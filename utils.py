import logging
import time
import pickle


def get_logger(config):
    pathname = "./log/{}_{}_{}_{}_{}_{}_{}_{}.txt".format(config.dataset, config.seed, config.dilation, config.conv_hid_size, config.rounds, config.batch_size, config.alpha, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index):
    text = "-".join([str(i) for i in index])
    return text

def decode(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {}
        head_dict = {}

        # # T,L
        for i in range(l):
            for j in range(l):
                for f in range(4):
                    if instance[i, j, f] > 0:
                        if f == 0 and j > i:
                            if instance[j, i, 1] > 0:
                                if i not in forward_dict:
                                    forward_dict[i] = [j]
                                else:
                                    forward_dict[i].append(j)
                                forward_dict[i] = list(set(forward_dict[i]))  
                        elif f == 2 and j >= i:
                            if i not in head_dict:
                                head_dict[i] = {j}
                            else:
                                head_dict[i].add(j)
                        elif f == 3 and j <= i:
                            if j not in head_dict:
                                head_dict[j] = {i}
                            else:
                                head_dict[j].add(i)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()
 
        for head in head_dict:
            find_entity(head, [], head_dict[head])


        predicts = set([convert_index_to_text(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_r, ent_p, ent_c

def decode_without_disconnect(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):

        ht_type_dict = {}
        predicts = []
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    predicts.append(list(range(i, j + 1)))

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_r, ent_p, ent_c