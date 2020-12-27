import os
import json
import pandas as pd
import tensorflow as tf


class DataLoader:
    """
    The data should be put under the path "Data/", and name the dataset with "data_name".
    Data files list in the path "Data/xxx":
    - train.json
    - test.json
    """
    def __init__(self, data_name):
        self.dataname = data_name
        self.datadir = os.path.join("Data", data_name)
        self.sentence = []

    def load_directory_data(self, file_path):
        data = {}
        data['tokens'] = []
        data['mentions'] = []
        if not os.path.isfile(file_path):
            print("[ERROR] No such file found.")
            return
        with tf.io.gfile.GFile(file_path, "r") as f:
            for line in f:
                line = line.strip()
                json_line = json.loads(line)
                data['tokens'].append(json_line['tokens'])
                data['mentions'].append(json_line['mentions'])
        return pd.DataFrame.from_dict(data)

    def load_dataset(self):
        train_df = self.load_directory_data(os.path.join(self.datadir, 'train.json'))
        test_df = self.load_directory_data(os.path.join(self.datadir, 'test.json'))
        return train_df, test_df
