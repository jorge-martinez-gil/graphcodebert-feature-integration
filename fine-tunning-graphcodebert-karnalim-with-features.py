# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024c] Improving Source Code Similarity Detection with GraphCodeBERT and Additional Feature Integration, arXiv preprint arXiv:xxxx.xxxxx, 2024

@author: Jorge Martinez-Gil
"""

# Install the required transformers package with PyTorch support
# The -U flag ensures that the package is updated to the latest version if not already installed.
#!pip install transformers[torch] -U

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Garbage collection to free up memory
import gc  
torch.cuda.empty_cache()  # Clear GPU memory
gc.collect()  # Collect any unused objects in CPU RAM

# Custom model class inheriting from nn.Module
class RobertaForSequenceClassificationWithOutput(nn.Module):
    def __init__(self, num_labels=2, output_feature_dim=1):
        """
        Initialize the model with:
        - num_labels: Number of output labels/classes for classification
        - output_feature_dim: Dimension of the additional output feature
        """
        super().__init__()
        self.num_labels = num_labels
        
        # Load the pre-trained GraphCodeBERT model
        self.roberta = AutoModel.from_pretrained('microsoft/graphcodebert-base')
        
        # Layer to process the additional output feature
        self.output_feature_layer = nn.Linear(output_feature_dim, self.roberta.config.hidden_size)
        
        # Classifier layer which concatenates the BERT output and additional feature
        self.classifier = nn.Linear(self.roberta.config.hidden_size + self.roberta.config.hidden_size, num_labels)
        
        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, labels=None, output_feature=None):
        """
        Forward pass of the model.
        """
        # Get the outputs from the BERT model
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Process the additional feature and concatenate with the BERT output
        output_feature_processed = self.output_feature_layer(output_feature.unsqueeze(-1))
        combined_features = torch.cat((pooled_output, output_feature_processed), dim=1)
        combined_features = self.dropout(combined_features)

        # Pass through the classifier
        logits = self.classifier(combined_features)

        # Calculate the loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return the output in the expected format
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Custom dataset class for code pairs
class CodePairDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        """
        Initialize the dataset with:
        - file_path: Path to the JSON dataset file
        - tokenizer: Pre-trained tokenizer for encoding the data
        """
        # Load the dataset from the file
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        """
        Get an item (code pair) from the dataset.
        """
        item = self.data[idx]
        
        # Tokenize the code pair
        encoding = self.tokenizer(text=item["code1"], text_pair=item["code2"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        
        # Squeeze the batch dimension out of the encoding
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}  
        
        # Add the label and the additional output feature
        encoding['labels'] = torch.tensor(item["score"], dtype=torch.long)
        encoding['output_feature'] = torch.tensor(item["output"], dtype=torch.float)
        
        return encoding

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.data)

# Function to compute evaluation metrics during training
def compute_metrics(p):
    """
    Compute accuracy, precision, recall, and F1 score for the predictions.
    """
    predictions, labels = p
    predictions = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'accuracy': accuracy_score(labels, predictions), 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    """
    Main function to run the training and evaluation.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
    
    # Path to the dataset file
    dataset_path = 'data\data2.json'  # Your dataset path
    
    # Load the dataset
    full_dataset = CodePairDataset(file_path=dataset_path, tokenizer=tokenizer)

    # Split dataset into training, validation, and test sets
    train_size = int(0.8 * len(full_dataset))
    test_val_size = len(full_dataset) - train_size
    val_size = int(0.5 * test_val_size)  # Half of the remaining for validation
    test_size = test_val_size - val_size

    # Randomly split the dataset into train, validation, and test sets
    train_dataset, remaining_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_val_size])
    val_dataset, test_dataset = torch.utils.data.random_split(remaining_dataset, [val_size, test_size])

    # Initialize the model
    model = RobertaForSequenceClassificationWithOutput(num_labels=2, output_feature_dim=1)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # Directory to save the model and results
        num_train_epochs=3,  # Number of epochs to train
        per_device_train_batch_size=8,  # Batch size per GPU/CPU
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Weight decay for regularization
        logging_dir='./logs',  # Directory for storing logs
        evaluation_strategy="steps",  # Evaluate the model every N steps
        eval_steps=500,  # Steps interval for evaluation
        save_strategy="steps",  # Save the model every N steps
        save_steps=500,  # Steps interval for saving the model
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="f1",  # Metric to use for selecting the best model
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Function to compute metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop early if no improvement
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation set
    val_results = trainer.evaluate(val_dataset)
    print(f"Validation Precision: {val_results['eval_precision']:.4f}")
    print(f"Validation Recall: {val_results['eval_recall']:.4f}")
    print(f"Validation F1 Score: {val_results['eval_f1']:.4f}")

    # Evaluate the model on the test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Test Recall: {test_results['eval_recall']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
