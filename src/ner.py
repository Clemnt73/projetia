from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

from typing import List


class NaturalEntityRecognizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.__tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._pipeline = pipeline(
            task='ner',
            model=self.__model,
            tokenizer=self.__tokenizer,
            aggregation_strategy='simple'
        )

    def __call__(self, text: str) -> List[str]:
        return [
            element['word']
            for element in self._pipeline(text)
        ]


if __name__ == '__main__':
    model = NaturalEntityRecognizer('Clemnt73/RoBERTa-ner')
    sentences = [
        "Qui est Emmanuel Macron ?",
        "Quelle est la capitale de la belgique ?",
        "Quelle est la superficie de la région urbaine de lyon ?"
    ]

    for sentence in sentences:
        print(sentence, '-->', model(sentence))
