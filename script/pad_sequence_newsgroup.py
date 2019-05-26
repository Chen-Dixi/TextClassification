import torch.nn.utils.rnn.pad_sequence as pad_sequence

#pad_sequence(sequences, batch_first=False, padding_value=0)
#sequences 是一个tensor
TEXT_DATA_DIR='data/20_newsgroup'
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath)
                texts.append(f.read())
                f.close()
                labels.append(label_id)

