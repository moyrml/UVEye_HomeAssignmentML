import pandas as pd


class LabelEncoder:
    def __init__(self, types):
        self.class_id_to_label = {i: j for i, j in enumerate(types)}
        self.label_to_class_id = {j: i for i, j in enumerate(types)}

    def encode(self, label):
        """
        :param label: str. type of item
        """
        return self.label_to_class_id[label]

    def decode(self, class_id):
        """
        :param class_id: int. Number representing item type
        """

        return self.class_id_to_label[class_id]

    def describe(self):
        return pd.DataFrame.from_dict(self.label_to_class_id, orient='index', columns=['Class ID'])