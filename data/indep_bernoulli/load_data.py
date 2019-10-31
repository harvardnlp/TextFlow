import os
import sys

from collections import namedtuple
from six.moves.urllib.request import urlopen

import six.moves.cPickle as pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

dset = namedtuple("dset", ["name", "url", "filename"])

JSB_CHORALES = dset("jsb_chorales",
                    "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle",
                    "jsb_chorales.pkl")

PIANO_MIDI = dset("piano_midi",
                  "http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle",
                  "piano_midi.pkl")

MUSE_DATA = dset("muse_data",
                 "http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle",
                 "muse_data.pkl")

NOTTINGHAM = dset("nottingham",
                  "http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle",
                  "nottingham.pkl")
str2obj = {'jsb_chorales': JSB_CHORALES, 'piano_midi': PIANO_MIDI, 'muse_data': MUSE_DATA, 'nottingham': NOTTINGHAM}


# this function processes the raw data; in particular it unsparsifies it
def process_data(base_path, dataset, min_note=21, note_range=88):
    output = os.path.join(base_path, dataset.filename)
    if os.path.exists(output):
        try:
            with open(output, "rb") as f:
                return pickle.load(f)
        except (ValueError, UnicodeDecodeError):
            # Assume python env has changed.
            # Recreate pickle file in this env's format.
            os.remove(output)

    print("processing raw data - {} ...".format(dataset.name))
    data = pickle.load(urlopen(dataset.url))
    processed_dataset = {}
    for split, data_split in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]['sequence_lengths'] = torch.zeros(n_seqs, dtype=torch.long)
        processed_dataset[split]['sequences'] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]['sequence_lengths'][seq] = seq_length
            processed_sequence = torch.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = torch.tensor(list(data_split[seq][t]), dtype=torch.int64) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_sequence[t, note_slice] = torch.ones(slice_length)
            processed_dataset[split]['sequences'].append(processed_sequence)
        print(split)
        print(n_seqs)
        print(processed_dataset[split]['sequence_lengths'])
        print(processed_dataset[split]['sequence_lengths'].max())
        print(processed_dataset[split]['sequences'][0][0], processed_dataset[split]['sequences'][0].shape)
    pickle.dump(processed_dataset, open(output, "wb"), pickle.HIGHEST_PROTOCOL)
    print("dumped processed data to %s" % output)


# this logic will be initiated upon import
base_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(base_path):
    os.mkdir(base_path)

# ingest training/validation/test data from disk
def load_data(dataset):
    # download and process dataset if it does not exist
    dataset = str2obj[dataset]
    process_data(base_path, dataset)
    file_loc = os.path.join(base_path, dataset.filename)
    with open(file_loc, "rb") as f:
        dset = pickle.load(f)
        #for k, v in dset.items():
        #    sequences = v["sequences"]
        #    dset[k]["sequences"] = pad_sequence(sequences, batch_first=True).type(torch.Tensor)
        #    dset[k]["sequence_lengths"] = v["sequence_lengths"]
    return dset

if __name__ == '__main__':
    for k, v in str2obj.items():
        load_data(k)
