import re


import string
from torcheval.metrics.functional import bleu_score
from collections import Counter

import torch

def process_string(s):
        # Convert to lowercase
        s = s.lower()
        # Remove all whitespace characters except spaces
        s = re.sub(r'[^\S ]+', '', s)
        # Replace multiple spaces with a single space
        s = re.sub(r' +', ' ', s)

         # Remove punctuation
        s = ''.join([char for char in s if char not in string.punctuation])

        return s.strip()  # Optionally, remove leading/trailing spaces


def calc_loss_and_metrics(predicted, target, tokenizer, print_=False):

    accuracy = 0
    bleu_scores = []
    precisions = []
    recalls = []
    f1_scores = []
    if print_:
        print(predicted)

    # Iterate over each sample in the batch
    for i in range(len(target)):  # Assuming target has shape (batch_size, ...)
        # Extract the individual target and predicted for the current batch item
        target_item = target[i]
        predicted_item = predicted[i]
        
        # Ensure the answer has its capitals and whitespace removed
        predicted_string = process_string(tokenizer.decode(predicted_item.long(), skip_special_tokens=True))
        target_string = process_string(tokenizer.decode(target_item.long(), skip_special_tokens=True))
        if print_:
            print("Predicted:")
            print(predicted_string)
            print("Answer:")
            print(target_string)

        predicted_list = predicted_string.split()
        target_list = target_string.split()

        if print_:
            print(predicted_list)
            print(target_list)

        if predicted_list == target_list:
            accuracy += 1.0

        if len(target_list) == 0 or len(predicted_list) == 0:
            bleu_score_ = 0.0
        else:
            bleu_score_ = bleu_score(predicted_string, [target_string], n_gram=1).item()

        # Calculate precision and recall
        prec = 0.0
        rec = 0.0
        if len(predicted_string) != 0 and len(target_string) != 0:
            predicted_count = Counter(predicted_list)
            target_count = Counter(target_list)

          # Calculate the number of common tokens (one-to-one matching)
            common_count = 0
            for token in predicted_count:
                if token in target_count:
                    # Take the minimum of the occurrences in both lists for exact matches
                    common_count += min(predicted_count[token], target_count[token])
            if print_:
                print(common_count)
            # Precision: proportion of predicted tokens that are correct
            prec = common_count / len(predicted_list)
            
            # Recall: proportion of target tokens that were predicted
            rec = common_count / len(target_list)



        if prec + rec == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)

        bleu_scores.append(bleu_score_)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    # Average the accuracy, BLEU scores, precision, recall, and F1 score over the batch
    accuracy = accuracy / len(target)  # Average accuracy
    average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    average_prec = sum(precisions) / len(precisions) if precisions else 0.0
    average_rec = sum(recalls) / len(recalls) if recalls else 0.0
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


    return {
        "acc": accuracy,
        "prec": average_prec,
        "recall": average_rec,
        "f1": average_f1,
        "bleu": average_bleu_score,
    }


class embeddings_metrics:
    def __init__(self, embedding: torch.Tensor):
        """
        Initialize with either an embedding tensor or by providing embedding_size (and optionally histogram and size).

        Args:
            embedding (torch.Tensor): A 3D tensor (batch_size, token_length, embedding_size).
            embedding_size (int): The size of the embeddings (required if embedding is not provided).
            histogram (torch.Tensor): Pre-computed histogram.
            size (int): The current size (number of tokens processed).
        """
        if embedding is not None:
            if len(embedding.size()) != 3:
                raise ValueError("Embedding tensor must be 3D (b, token_length, embedding_size)")
            self.embedding_size = embedding.size(2)
            self.histogram = torch.zeros(self.embedding_size)
            self.size = 0
            self.iterate_histogram(embedding)
        else:
            raise ValueError("Either an embedding tensor or embedding_size must be provided.")

    def iterate_histogram(self, embeddings):
        # embeddings is expected to be a 3D tensor: (batch_size, token_length, embedding_size)
        if embeddings.size(2) != self.embedding_size:
            raise ValueError(
                f"Embedding dim {embeddings.size(2)} does not match the histogram's embedding size {self.embedding_size}"
            )
        # Sum over the batch and token dimensions, preserving the embedding dimension
        self.histogram += embeddings.sum(dim=(0, 1))
        # Update size to reflect the total number of tokens processed
        self.size += embeddings.size(0) * embeddings.size(1)

    def __add__(self, other):
        if not isinstance(other, embeddings_metrics):
            raise TypeError("Can only add embeddings_metrics objects")
        

        print(other.histogram.size(), self.histogram.size())

        new_histogram = self.histogram + other.histogram
        new_size = self.size + other.size

        print(new_histogram.size())

        new_obj = embeddings_metrics(torch.zeros([1,1,self.embedding_size]))




        new_obj.histogram = new_histogram
        new_obj.size = new_size

        return new_obj

    def __sub__(self, other):
        if not isinstance(other, embeddings_metrics):
            raise TypeError("Can only subtract embeddings_metrics objects")
        new_histogram = self.histogram - other.histogram
        new_size = self.size - other.size

        new_obj = embeddings_metrics(torch.zeros([1,1,new_histogram.size(0)]))
        new_obj.histogram = new_histogram
        new_obj.size = new_size

        return new_obj

    def __truediv__(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Can only divide embeddings_metrics by an int or float")
        new_histogram = self.histogram / value
        new_size = self.size / value

        new_obj = embeddings_metrics(torch.zeros([1,1,new_histogram.size(0)]))
        new_obj.histogram = new_histogram
        new_obj.size = new_size

        return new_obj

    def get_histogram(self):
        # Returns the average histogram if tokens have been processed, else the raw histogram.
        if self.size == 0:
            return self.histogram
        return self.histogram / self.size

    def size(self):
        return self.histogram.size()
    
class Metrics:
    def __init__(self, 
                 original_embedding=embeddings_metrics(torch.zeros([1,1,10])), 
                 restored_projected_embedding=embeddings_metrics(torch.zeros([1,1,10])), 
                 projected_embedding=embeddings_metrics(torch.zeros([1,1,10])), 
                 loss=0, 
                 token_prediction_loss=0, 
                 regularisation_loss=0, 
                 acc=0, 
                 prec=0, 
                 recall=0, 
                 f1=0, 
                 bleu=0):
        
        embeddings_metrics(torch.zeros([1,1,10]))
        self.metrics = {
            "loss": loss,
            "token_prediction_loss": token_prediction_loss,
            "regularisation_loss": regularisation_loss,
            "acc": acc,
            "prec": prec,
            "recall": recall,
            "f1": f1,
            "bleu": bleu,
            "original_embedding": original_embedding,
            "restored_projected_embedding": restored_projected_embedding,
            "projected_embedding": projected_embedding
        }

    def __add__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError("Can only add Metrics objects")
        
        print(other.metrics["original_embedding"].size(), self.metrics["original_embedding"].histogram.size())
        
        return Metrics(
            loss=self.metrics["loss"] + other.metrics["loss"],
            token_prediction_loss=self.metrics["token_prediction_loss"] + other.metrics["token_prediction_loss"],
            regularisation_loss=self.metrics["regularisation_loss"] + other.metrics["regularisation_loss"],
            acc=self.metrics["acc"] + other.metrics["acc"],
            prec=self.metrics["prec"] + other.metrics["prec"],
            recall=self.metrics["recall"] + other.metrics["recall"],
            f1=self.metrics["f1"] + other.metrics["f1"],
            bleu=self.metrics["bleu"] + other.metrics["bleu"],
            original_embedding=self.metrics["original_embedding"] + other.metrics["original_embedding"],
            restored_projected_embedding=self.metrics["restored_projected_embedding"] + other.metrics["restored_projected_embedding"],
            projected_embedding=self.metrics["projected_embedding"] + other.metrics["projected_embedding"]
        )

    def __sub__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError("Can only subtract Metrics objects")
        return Metrics(
            loss=self.metrics["loss"] - other.metrics["loss"],
            token_prediction_loss=self.metrics["token_prediction_loss"] - other.metrics["token_prediction_loss"],
            regularisation_loss=self.metrics["regularisation_loss"] - other.metrics["regularisation_loss"],
            acc=self.metrics["acc"] - other.metrics["acc"],
            prec=self.metrics["prec"] - other.metrics["prec"],
            recall=self.metrics["recall"] - other.metrics["recall"],
            f1=self.metrics["f1"] - other.metrics["f1"],
            bleu=self.metrics["bleu"] - other.metrics["bleu"],
            original_embedding=self.metrics["original_embedding"] - other.metrics["original_embedding"],
            restored_projected_embedding=self.metrics["restored_projected_embedding"] - other.metrics["restored_projected_embedding"],
            projected_embedding=self.metrics["projected_embedding"] - other.metrics["projected_embedding"]
        )

    def __truediv__(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Can only divide Metrics by an int or float")
        return Metrics(
            loss=self.metrics["loss"] / value,
            token_prediction_loss=self.metrics["token_prediction_loss"] / value,
            regularisation_loss=self.metrics["regularisation_loss"] / value,
            acc=self.metrics["acc"] / value,
            prec=self.metrics["prec"] / value,
            recall=self.metrics["recall"] / value,
            f1=self.metrics["f1"] / value,
            bleu=self.metrics["bleu"] / value,
            original_embedding=self.metrics["original_embedding"] / value,
            restored_projected_embedding=self.metrics["restored_projected_embedding"] / value,
            projected_embedding=self.metrics["projected_embedding"] / value
        )

    def get_log(self, header=""):
        """
        Returns a dictionary of metrics with an optional header prepended to each key.
        For the embedding histograms, returns the actual histogram tensors.

        Args:
            header (str): A string to prepend to each key.

        Returns:
            dict: A dictionary with updated keys.
        """
        result = {}
        for key, value in self.metrics.items():
            if key in ["original_embedding", "restored_projected_embedding", "projected_embedding"]:
                # Special case for embedding histograms: get the actual histogram tensor
                result[f"{header}{key}"] = value.get_histogram()
            else:
                result[f"{header}{key}"] = value
        return result

    def __repr__(self):
        return f"Metrics({self.metrics})"
    
class MetricsList(list):
    def __init__(self, *metrics):
        # Ensure all elements are Metrics instances
        for metric in metrics:
            if not isinstance(metric, Metrics):
                raise TypeError("All elements must be Metrics objects.")
        super().__init__(metrics)

    def append(self, metric):
        if not isinstance(metric, Metrics):
            raise TypeError("Only Metrics objects can be appended.")
        super().append(metric)

    def extend(self, metrics):
        if not all(isinstance(metric, Metrics) for metric in metrics):
            raise TypeError("Only Metrics objects can be added.")
        super().extend(metrics)

    def total(self):
        """Compute the total (sum) of all Metrics in the list."""
        if not self:
            return Metrics()  # Return an empty Metrics object if the list is empty
        total = self[0]
        for metric in self[1:]:
            total += metric
        return total

    def average(self):
        """Compute the average of all Metrics in the list."""
        if not self:
            return Metrics()  # Return an empty Metrics object if the list is empty
        total = self.total()
        return total / len(self)

    def __add__(self, other):
        """Add corresponding Metrics objects in the list."""
        if not isinstance(other, MetricsList):
            raise TypeError("Can only add another MetricsList.")
        if len(self) != len(other):
            raise ValueError("Both MetricsLists must have the same length.")

        # Add corresponding Metrics in both lists
        result = MetricsList(*(self[i] + other[i] for i in range(len(self))))
        return result

    def __truediv__(self, value):
        """Divide each Metrics object in the list by a scalar value."""
        if not isinstance(value, (int, float)):
            raise TypeError("Can only divide by an integer or float.")

        result = MetricsList(*(metric / value for metric in self))
        return result

    def __repr__(self):
        return f"MetricsList({super().__repr__()})"
