import pandas as pd
import os.path as osp
import os
import re
import string

"""This class splits the dataset into 5K sample partitions after preprocess operation."""
class SplitDataset():
    def __init__(self, dataset_root_path):
        self.dataset_root_path = dataset_root_path

    def split(self):
        for i in range(2, 4):
            dataset_path = osp.join(self.dataset_root_path, str(i))
            data_path = osp.join(dataset_path, [dir for dir in os.listdir(dataset_path) if dir.find(".csv") != -1][0])

            self.dataset = pd.read_csv(data_path)
            self.preprocess()
            dataset_range = len(self.dataset) // 5000
            os.makedirs(osp.join(dataset_path, "Splits"), exist_ok=True)
            for k in range(dataset_range + 1):
                if k == 0:
                    split = self.dataset.iloc[5000*k : 5000*(k+1), :]
                    split.to_csv(osp.join(dataset_path, "Splits",  "Train_" + str((k+1) * 5000) + ".csv"), index = False)
                else:
                    split = self.dataset.iloc[5000 : 5000 * (k + 1), :]
                    split.to_csv(osp.join(dataset_path, "Splits", str((k) * 5000) + ".csv"), index=False)
            print("Dataset : ", i, " ready..")

    def preprocess(self):
        # Deletes the duplicated rows if exists
        self.dataset.drop_duplicates(inplace=True)
        contraction_dict = {"ain't": "are not", "'s": " is", "aren't": "are not", "isn't" : "is not", "'m" : " am", "didn't" : "did not", "don't" : "do not",}
        contractions_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))

        def expand_contractions(text, contractions_dict=contraction_dict):
            def replace(match):
                return contractions_dict[match.group(0)]

            return contractions_re.sub(replace, text)

        # Expands the contradictions
        self.dataset['source_text'] = self.dataset['source_text'].apply(lambda x: expand_contractions(x))
        self.dataset['target_text'] = self.dataset['target_text'].apply(lambda x: expand_contractions(x))

        # Make all the words lowercase format // Optional
        # self.dataset["source_text"] = self.dataset["source_text"].str.lower()
        # self.dataset["target_text"] = self.dataset["target_text"].str.lower()

        # Deletes the punctuations
        self.dataset['source_text'] = self.dataset['source_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
        self.dataset['target_text'] = self.dataset['target_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

        # Delete multiple spaces
        self.dataset["source_text"] = self.dataset["source_text"].apply(lambda x: re.sub(' +', ' ', x))
        self.dataset["target_text"] = self.dataset["target_text"].apply(lambda x: re.sub(' +', ' ', x))

if __name__ == "__main__":
    split_dataset = SplitDataset("/home/onuraydemir/Desktop/Kolektif Proje/Datasets")
    split_dataset.split()
