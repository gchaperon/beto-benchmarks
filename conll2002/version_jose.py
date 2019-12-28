#!/usr/bin/env python
# coding: utf-8

# # Evaluation of BERT on NER (CoNLL-2002)

# ## Notebook parameters

# In[3]:


SEED = 1234
init_checkpoint_dir = '/data/gchaperon/beto-cased'
#init_checkpoint_dir = '../spanish_L-12_H-768_A-12/pytorch'
device_to_use = 'cuda' # 'cpu' or 'cuda'
batch_size = 8
learning_rate = 2e-5
output_dir = './output-cased'
train_file = '/data/gchaperon/conll2002/esp.train'
dev_file = '/data/gchaperon/conll2002/esp.testa'
formatted_train_file = './formatted_train.csv'
formatted_dev_file = './formatted_dev.csv'
train_ratio = 0.8
training_epochs = 2


# In[4]:

from tqdm import tqdm
import torch
import time
import random
import os
from pytorch_transformers import BertForTokenClassification, BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from torchtext import data
from torchtext import datasets
from math import ceil


# In[5]:


torch.manual_seed(SEED)
device = torch.device(device_to_use)


# ## Set-up Trains

# In[6]:


#from trains import Task
#task = Task.init(project_name="POS Evaluation", task_name="POS Evaluation")


# ## Data Preparation

# In[7]:


import pyconll
import csv
import unicodedata
from torch.utils.data import Dataset


def is_number(s):
    try:
        float(s.replace(',', '.'))
        return True
    except ValueError:
        return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def remove_splitting_symbols(text):
    text = text.replace('...', '.')
    if len(text) > 1:
        for char in text:
            if is_punctuation(char) and len(text) > 1:
                text = text.replace(char, '', 1)
    return text

def convert_to_single_number(text):
    if is_number(text):
        return text[0]
    return text
    
def process_token(token):
    token = remove_splitting_symbols(token)
    token = convert_to_single_number(token)
    return token
    
def convert_ner_to_dataset(original_file, formatted_file):
    with open(original_file, 'r') as reader:
        with open(formatted_file, 'w') as writing_file:
            writing_file = csv.writer(writing_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writing_file.writerow(['text', 'labels', 'masks'])
            text = ''
            labels = ''
            for line in reader:
                if len(line.split()) < 3:
                    if len(text.split()) > 0 and len(text.split()) < 480:
                        writing_file.writerow([text, labels, text])
                    else:
                        print('acá')
                    text = ''
                    labels = ''
                else:
                    splitted = line.split()
                    text += process_token(splitted[0]) + ' '
                    labels += splitted[2] + ' '


# In[8]:


print('DEV SET')
convert_ner_to_dataset(dev_file, formatted_dev_file)
print('TRAIN SET')
convert_ner_to_dataset(train_file, formatted_train_file)


# In[7]:


tokenizer = BertTokenizer(vocab_file=init_checkpoint_dir+'/vocab.txt', do_lower_case=False, do_basic_tokenize=True)
#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[8]:


def bert_tokenizer_for_POS(text):
    #text = text.lower()
    text_tokens = tokenizer.tokenize(text)
    ids_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
    return ids_tokens

def tokenizer_for_masking(text):
    #text = text.lower()
    text_tokens = tokenizer.tokenize(text)
    return [0 if '##' in token else 1 for token in text_tokens]


# In[9]:


TEXT = data.Field(tokenize=bert_tokenizer_for_POS, use_vocab=False, lower=False, unk_token=3, pad_token=1)
LABELS = data.Field(is_target=True, unk_token='<pad>', pad_token='<pad>')
MASKS = data.Field(tokenize=tokenizer_for_masking, use_vocab=False, lower=False, unk_token=3, pad_token=0)


fields = {'text': ('text', TEXT), 'labels': ('labels', LABELS), 'masks': ('masks', MASKS)}

train_dataset = data.TabularDataset(
    path=formatted_train_file,
    format='csv',
    fields=fields
)

dev_dataset = data.TabularDataset(
    path=formatted_dev_file,
    format='csv',
    fields=fields
)


LABELS.build_vocab({'B-LOC': 4, 'B-MISC': 8, 'I-ORG': 3, 'O': 1, 'I-PER': 6, 'I-MISC': 7, 'I-LOC': 0, 'B-ORG': 2, 'B-PER': 5, '<pad>': 9})
LABELS.vocab.stoi = {'B-LOC': 4, 'B-MISC': 8, 'I-ORG': 3, 'O': 1, 'I-PER': 6, 'I-MISC': 7, 'I-LOC': 0, 'B-ORG': 2, 'B-PER': 5, '<pad>': 9}
LABELS.vocab.itos = ['I-LOC', 'O', 'B-ORG', 'I-ORG', 'B-LOC', 'B-PER', 'I-PER', 'I-MISC', 'B-MISC', '<pad>']


# In[10]:


print(train_dataset[0].masks)
print(train_dataset[0].text)
print(len(train_dataset[0].masks))
print(len(train_dataset[0].text))
print(len(train_dataset[0].labels))


# ## Classification

# In[11]:


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


# In[12]:


def evaluation(model, iterator, loss_function):
    
    model.eval()
    
    test_accuracy = 0
    test_loss = 0
    
    with torch.no_grad():
        for batch in iterator:

            x = torch.t(batch.text)
            y = torch.t(batch.labels)
            masks = torch.t(batch.masks)

            y_logits = model(x)
            y_logits = y_logits[0]
            
            masks = masks.expand(len(LABELS.vocab) - 1, masks.shape[0], masks.shape[1])
            masks = masks.permute(1, 2, 0)

            y_logits = y_logits * masks.float()

            y_logits = y_logits[y_logits.sum(2)!=0]

            y = y.reshape(-1)
            y = y[y!=LABELS.vocab.stoi['<pad>']] # 17 is pad token

            L = loss_function(y_logits, y).item()

            test_loss += L
            test_accuracy += categorical_accuracy(y_logits, y).item() * 100
        
    return test_loss / len(iterator), test_accuracy / len(iterator)
    

def train(model, iterator, scheduler, optim, batch_size, loss_function, run_in_GPU=False):
                
    model.train()
    
    train_acc = 0
    train_loss = 0
    
    for batch in tqdm(iterator, desc="Train"):
        # breakpoint()
        # Vacía los gradientes.
        model.zero_grad()
        # Vaciar los gradientes es muy importante pues pytorch nos permite
        # tener control total sobre los gradientes que vamos computando y
        # por ejemplo acumularlos desde distintas redes.

        x = torch.t(batch.text)
        y = torch.t(batch.labels)
        masks = torch.t(batch.masks)
        
        #print(y)
        #print(masks)
        
        #y[y==0] = 18 # cambio el label  
        #y[y==LABELS.vocab.stoi['<pad>']] = 0
        
        # Usa la red para computar la predicción.
        y_logits = model(x)
        y_logits = y_logits[0]
        
        masks = masks.expand(len(LABELS.vocab) - 1, masks.shape[0], masks.shape[1])
        masks = masks.permute(1, 2, 0)
        
        y_logits = y_logits * masks.float()

        y_logits = y_logits[y_logits.sum(2)!=0]
               
        y = y.reshape(-1)
        y = y[y!=LABELS.vocab.stoi['<pad>']] # 17 is pad token
        
        #print(y.shape)
        #print(y_logits.shape)
        
        # Calcula la función de pérdida.
        L = loss_function(y_logits, y)


        #print('Loss: ' + str(L.item()))
        
        train_loss += L.item()
        train_acc += categorical_accuracy(y_logits, y).item() * 100

        # Computa la pasada hacia atrás.
        L.backward()

        # Computa un paso del optimizador (modifica los pesos).
        scheduler.step()
        optim.step()
        
    # Listo! :-)
        
    return train_loss / len(iterator), train_acc / len(iterator)


# In[13]:


def run_classifier(model, train_dataset, test_dataset, output_dir='./models/', split_ratio=0.7, loss_function=torch.nn.CrossEntropyLoss(), lr=0.01, batch_size=8, epochs=10, device=None):
    
    best_valid_loss = float('inf')
    
    train_iterator = data.Iterator(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    test_iterator = data.Iterator(test_dataset, batch_size=batch_size, device=device)

    optimization_steps = epochs * ceil(len(train_dataset) / batch_size)
    
    optimizador = AdamW(params=model.parameters(), lr=lr, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizador, warmup_steps=0.1*optimization_steps, t_total=optimization_steps)

    
    for epoch in range(1, epochs+1):
                
        train_loss, train_acc = train(model, train_iterator, scheduler, optimizador, batch_size, loss_function)
    
        test_loss, test_acc = evaluation(model, test_iterator, loss_function)
        
        if test_loss < best_valid_loss:
            best_valid_loss = test_loss

            # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
            model_to_save = model.module if hasattr(model, 'module') else model

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)
        
        print('\rEpoch: {} Train loss: {:.3f} Train accuracy: {:.3f}% Validation loss: {:.3f}, Validation Accuracy: {:.3f}%'.format(
            epoch, train_loss, train_acc, test_loss, test_acc))
        
    return model


# In[14]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[15]:


model = BertForTokenClassification.from_pretrained(init_checkpoint_dir, num_labels=len(LABELS.vocab)-1)
#model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(LABELS.vocab)-1)
model.to(device)


# In[16]:


print('parametros a optimizar: ' + str(count_parameters(model)))


# In[17]:


run_classifier(model, train_dataset, dev_dataset, epochs=training_epochs, batch_size=batch_size, lr=learning_rate, split_ratio=train_ratio, output_dir=output_dir, device=device)
breakpoint()

# In[18]:


print(LABELS.vocab.stoi)
print(LABELS.vocab.itos)


# In[ ]:




