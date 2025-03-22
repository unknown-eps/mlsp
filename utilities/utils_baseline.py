# This file is used to build utilities for the baseline model

from transformers import DistilBertTokenizerFast
from datasets import load_dataset
import evaluate
class BaseUtilsImdb:
    '''
    This class is used to build utilities for the baseline model
    The split defines the percentage of the data to be used for validation from the training data
    '''
    def __init__(self,split=0.1,random_seed=42):
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
    
    
    #TODO: Add the function to clean the text
    @classmethod
    def clean_text(cls,text):
        '''
        It removes the html tags from the text and replaces multiple spaces with a single space
        '''
        raise NotImplementedError     

        return text
    
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