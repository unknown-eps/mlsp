# This file is used to build utilities for the baseline model

from transformers import DistilBertTokenizerFast
from datasets import load_dataset
import evaluate
class BaseUtilsImdb:
    '''
    This class is used to build utilities for the baseline model
    The split defines the percentage of the data to be used for validation from the training data
    '''
    def __init__(self,split=0.1,random_seed=42,clean_text=False):
        '''
        It initializes the class with the tokenizer and the dataset and splits the data into training, validation and test data
        Stratify by column is used to ensure that the distribution of the labels is the same in the training and validation data
        '''
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dataset = load_dataset('imdb')
        self.random_seed = random_seed
        train_data = self.dataset['train']
        test_data = self.dataset['test']
        train_val_split = train_data.train_test_split(test_size=0.1, seed=self.random_seed, stratify_by_column='label')
        self.train_data = train_val_split['train']
        self.val_data = train_val_split['test']
        self.test_data = test_data
        if clean_text:
            self.train_data = self.train_data.map(lambda example: {'text': self.clean_text(example['text'])})
            self.val_data = self.val_data.map(lambda example: {'text': self.clean_text(example['text'])})
            self.test_data = self.test_data.map(lambda example: {'text': self.clean_text(example['text'])})
    
    def tokenize_function(self,examples):
        '''
        Tokenize the text column of the input examples
        Cuts the text if it is longer than the maximum length of the model
        The responsibilty of the padding is left to the dataloader
        '''
        return self.tokenizer(examples['text'], truncation=True)
    
    def get_tokenized_datasets(self):
        '''
        Tokenize the training, validation and test data
        '''
        tokenized_train_data = self.train_data.map(self.tokenize_function, batched=True)
        tokenized_val_data = self.val_data.map(self.tokenize_function, batched=True)
        tokenized_test_data = self.test_data.map(self.tokenize_function, batched=True)
        
        tokenized_train_data = tokenized_train_data.remove_columns(['text'])
        tokenized_val_data = tokenized_val_data.remove_columns(['text'])
        tokenized_test_data = tokenized_test_data.remove_columns(['text'])
        
        return tokenized_train_data, tokenized_val_data, tokenized_test_data
    
    
    def clean_text(self,text):
        '''
        It removes the html tags from the text and replaces multiple spaces with a single space
        '''
        import re
        from bs4 import BeautifulSoup
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        #Remove multiple spaces,newlines and tabs with a single space
        text = re.sub(r'\n\s*\n+', '\n', text) # Remove multiple newlines with a single newline
        text = re.sub(r'(?<=\S)[ ]{2,}', ' ', text)  # Ensures only consecutive spaces collapse

        return text.strip()
    
    @classmethod
    def compute_metric_accuracy(cls,eval_pred):
        '''
        It computes the accuracy of the model
        '''
        accuracy = evaluate.load('accuracy')
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return accuracy.compute(predictions=predictions,references=labels)
    
    @classmethod
    def write_time(cls,start_time,end_time,batch_size,epochs):
        '''
        It writes the time taken to train the model
        '''
        with open('time.txt','a+') as f:
            f.write(f'Batch size {batch_size}:epochs={epochs}:{int(end_time-start_time)} seconds\n')
            
            
class BaseUtilsCola:
    def __init__(self,random_seed=42):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dataset = load_dataset("glue", "cola")
        self.random_seed = random_seed
        self.train_data = self.dataset['train']
        self.val_data = self.dataset['validation']
        self.test_data = self.dataset['test']
        
    def tokenize_function(self,examples):
        '''
        Tokenize the text column of the input examples
        Cuts the text if it is longer than the maximum length of the model
        The responsibilty of the padding is left to the dataloader
        '''
        return self.tokenizer(examples['sentence'], truncation=True)
    
    def get_tokenized_datasets(self):
        '''
        Tokenize the training, validation and test data
        '''
        tokenized_train_data = self.train_data.map(self.tokenize_function, batched=True)
        tokenized_val_data = self.val_data.map(self.tokenize_function, batched=True)
        tokenized_test_data = self.test_data.map(self.tokenize_function, batched=True)
        
        tokenized_train_data = tokenized_train_data.remove_columns(['sentence'])
        tokenized_val_data = tokenized_val_data.remove_columns(['sentence'])
        tokenized_test_data = tokenized_test_data.remove_columns(['sentence'])
        
        return tokenized_train_data, tokenized_val_data, tokenized_test_data
    
    @classmethod
    def compute_metric_mcc(cls, eval_pred):
        '''
        Computes the Matthews correlation coefficient (MCC) for the CoLA dataset.
        '''
        matthews_corr = evaluate.load("matthews_correlation")
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return matthews_corr.compute(predictions=predictions, references=labels)
    
    @classmethod
    def write_time(cls,start_time,end_time,batch_size,epochs):
        '''
        It writes the time taken to train the model
        '''
        with open('time.txt','a+') as f:
            f.write(f'Batch size {batch_size}:epochs={epochs}:{int(end_time-start_time)} seconds\n')
            
    def save_predictions(self, predictions,name):
        import pandas as pd

        df = pd.DataFrame({'Label':predictions,"Id":list(range(1,len(predictions)+1))})
        df.to_csv(name+'.csv', index=False)
        