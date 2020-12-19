import json
import codecs
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import copy

def read_fspc(datapath):
    texts = []
    labels = []
    for line in codecs.open(datapath, encoding='utf8').readlines():
        dat = json.loads(line)
        texts.append(dat['poem'])
        labels.append(int(dat['setiments']['holistic'])-1)

    return texts, labels

all_texts, all_labels = read_fspc('./data/FSPC_V1.0.json')

from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=.1)
train_texts, dev_texts, train_labels, dev_labels = train_test_split(train_texts, train_labels, test_size=.2)

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

import torch

class FSPCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FSPCDataset(train_encodings, train_labels)
dev_dataset = FSPCDataset(dev_encodings, dev_labels)
test_dataset = FSPCDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
configuration = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=configuration)
model.to(device)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

best_model = None
best_acc = -1

for epoch in range(5):
    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

    model.eval()
    preds, targets = [], []
    for batch in dev_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        outputs = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=-1)
        targets.extend(labels.tolist())
        preds.extend(outputs.tolist())

    dev_accuracy = accuracy_score(y_true=targets, y_pred=preds)
    dev_precision_macro = precision_score(y_true=targets, y_pred=preds, average='macro')
    dev_precision_micro = precision_score(y_true=targets, y_pred=preds, average='micro')
    dev_recall_macro = recall_score(y_true=targets, y_pred=preds, average='macro')
    dev_recall_micro = recall_score(y_true=targets, y_pred=preds, average='micro')

    print("[Dev Accuracy]:", dev_accuracy)
    # print("[Dev Macro Precision]:", dev_precision_macro)
    # print("[Dev Micro Precision]:", dev_precision_micro)
    # print("[Dev Macro Recall:", dev_recall_macro)
    # print("[Dev Micro Recall]:", dev_recall_micro)

    if dev_accuracy > best_acc:
        best_acc = dev_accuracy
        best_model = copy.deepcopy(model)
        print("Saving the Best Model!")


model.eval()
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

test_preds, test_targets = [], []

for batch in test_loader:

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=-1)
    test_preds.extend(outputs.tolist())
    test_targets.extend(labels.tolist())

test_accuracy = accuracy_score(y_true=test_targets, y_pred=test_preds)
test_precision_macro = precision_score(y_true=test_targets, y_pred=test_preds, average='macro')
test_precision_micro = precision_score(y_true=test_targets, y_pred=test_preds, average='micro')
test_recall_macro = recall_score(y_true=test_targets, y_pred=test_preds, average='macro')
test_recall_micro = recall_score(y_true=test_targets, y_pred=test_preds, average='micro')

print("[Test Accuracy]:", test_accuracy)
print("[Test Macro Precision]:", test_precision_macro)
print("[Test Micro Precision]:", test_precision_micro)
print("[Test Macro Recall:", test_recall_macro)
print("[Test Micro Recall]:", test_recall_micro)