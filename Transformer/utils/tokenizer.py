import spacy


class Tokenizer(object):
    """
    Args in each tokenizer:
        text (str): raw sentence or texts

    Returns:
        tokens (list[str]): tokenized words from raw text

    Example:
        input: Hello, my name is jeffrey.
        output: ['Hello', ',', 'my', 'name', 'is', 'jeffrey', '.']
    """

    def __init__(self) -> None:
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(self, text):
        tokens = [tok.text for tok in self.spacy_de.tokenizer(text)]
        return tokens

    def tokenize_en(self, text):
        tokens = [tok.text for tok in self.spacy_en.tokenizer(text)]
        return tokens
