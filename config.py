import json


class Config:
    def __init__(self, args, is_bert=False):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        if is_bert:
            with open("./config/bert_config/config.json", "r") as f:
                bert_config = json.loads(f.read())
            for key in bert_config:
                if key == "hidden_size":
                    exec(f"self.{key} = config['lstm_hid_size']")
                elif key == "hidden_size":
                    exec(f"self.{key} = config['lstm_hid_size']")
                else:
                    exec(f"self.{key} = bert_config['{key}']")
        else:
            self.dataset = config["dataset"]
            self.dist_emb_size = config["dist_emb_size"]
            self.type_emb_size = config["type_emb_size"]
            self.lstm_hid_size = config["lstm_hid_size"]
            self.conv_hid_size = config["conv_hid_size"]
            self.bert_hid_size = config["bert_hid_size"]
            self.biaffine_size = config["biaffine_size"]
            self.ffnn_hid_size = config["ffnn_hid_size"]
            self.dilation = config["dilation"]

            self.emb_dropout = config["emb_dropout"]
            self.conv_dropout = config["conv_dropout"]
            self.out_dropout = config["out_dropout"]

            self.epochs = config["epochs"]
            self.batch_size = config["batch_size"]

            self.learning_rate = config["learning_rate"]
            self.weight_decay = config["weight_decay"]

            self.bert_name = config["bert_name"]
            self.bert_learning_rate = config["bert_learning_rate"]
            self.warm_factor = config["warm_factor"]

            self.use_bert_last_4_layers = config["use_bert_last_4_layers"]

            self.seed = config["seed"]
            self.rounds = config["rounds"]
            self.alpha = config["alpha"]
            self.cr = config["cr"]


            for k, v in args.__dict__.items():
                if v is not None:
                    self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())