import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import pandas as pd


class CommentDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
        label_to_id: Optional[Dict[str, int]] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id

        if self.labels is not None and self.label_to_id is not None:
            self.label_ids = [self.label_to_id[str(label)] for label in self.labels]
        elif self.labels is not None:
            self.label_ids = self.labels
        else:
            self.label_ids = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.label_ids is not None:
            item['labels'] = torch.tensor(self.label_ids[idx], dtype=torch.long)

        return item


class MultiTaskCommentDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        sentiment_labels: Optional[List] = None,
        toxicity_labels: Optional[List] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
        sentiment_label_to_id: Optional[Dict[str, int]] = None,
        toxicity_label_to_id: Optional[Dict[str, int]] = None
    ):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.toxicity_labels = toxicity_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentiment_label_to_id = sentiment_label_to_id
        self.toxicity_label_to_id = toxicity_label_to_id

        if self.sentiment_labels is not None and self.sentiment_label_to_id is not None:
            self.sentiment_label_ids = [
                self.sentiment_label_to_id[str(label)] for label in self.sentiment_labels
            ]
        elif self.sentiment_labels is not None:
            self.sentiment_label_ids = self.sentiment_labels
        else:
            self.sentiment_label_ids = None

        if self.toxicity_labels is not None and self.toxicity_label_to_id is not None:
            self.toxicity_label_ids = [
                self.toxicity_label_to_id[str(label)] for label in self.toxicity_labels
            ]
        elif self.toxicity_labels is not None:
            self.toxicity_label_ids = self.toxicity_labels
        else:
            self.toxicity_label_ids = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.sentiment_label_ids is not None:
            item['sentiment_labels'] = torch.tensor(
                self.sentiment_label_ids[idx],
                dtype=torch.long
            )

        if self.toxicity_label_ids is not None:
            item['toxicity_labels'] = torch.tensor(
                self.toxicity_label_ids[idx],
                dtype=torch.long
            )

        return item


def create_label_mappings(labels: List) -> tuple[Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(str(label) for label in labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def prepare_data_for_training(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    text_column: str = 'text_clean',
    label_column: str = 'sentiment',
    max_length: int = 128
) -> tuple[CommentDataset, Dict[str, int], Dict[int, str]]:
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    label_to_id, id_to_label = create_label_mappings(labels)

    dataset = CommentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id
    )

    return dataset, label_to_id, id_to_label

