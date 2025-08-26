import nltk
from nltk.corpus import shakespeare
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import pickle
import os


def extract_doc_text(text):
    # Remove XML headers
    text = re.sub(r"<\?xml.*?\?>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\!--.*?-->", "", text, flags=re.DOTALL)
    # Remove all XML/HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()  # keeps internal newlines; only trims ends


def get_shakespeare_doc_text(doc_name):
    """
    Returns the raw text of a specific Shakespeare document.
    """
    nltk.download("shakespeare", quiet=True)
    if doc_name in shakespeare.fileids():
        return extract_doc_text(shakespeare.raw(doc_name))
    else:
        raise ValueError(f"Document '{doc_name}' not found in NLTK Shakespeare corpus.")


# helper that splits on spaces/tabs ONLY (keeps '\n' inside tokens)
def split_on_spaces_only(s: str):
    tokens = re.split(r"[ \t]+", s)
    return [t for t in tokens if t != ""]


if __name__ == "__main__":
    DATA_DIR = "./data"
    os.makedirs(DATA_DIR, exist_ok=True)

    nltk.download("shakespeare", quiet=True)
    docs = shakespeare.fileids()
    print("Available Shakespeare documents:", docs)

    corpus = ""
    for doc in docs:
        text = get_shakespeare_doc_text(doc)
        corpus += text  # NEW: keep a newline between documents

    with open(os.path.join(DATA_DIR, "Shakespeare_clean_full.txt"), "w") as text_file:
        text_file.write(corpus)

    print("full_size: ", len(corpus))

    # compute word counts using spaces-only splitting (newlines preserved)
    total_words = len(split_on_spaces_only(corpus))
    test_size = total_words // 100
    valid_size = total_words // 100

    test_data = ""
    valid_data = ""
    random = random.Random()

    # normalize spaces/tabs but DO NOT touch newlines
    corpus = re.sub(r"[ \t]+", " ", corpus).strip(" ")  # keeps '\n' intact

    # Build test split by sampling contiguous spans in the space-split token view.
    for i in range(10):
        tokens = split_on_spaces_only(corpus)
        if len(tokens) <= test_size:
            break
        random_index = random.randint(0, len(tokens) - test_size)
        sub_test_text = " ".join(tokens[random_index : random_index + test_size])  # newlines inside tokens survive
        test_data += (" " if test_data else "") + sub_test_text
        corpus = corpus.replace(sub_test_text, "", 1)  # NEW: replace only first occurrence

    # build valid split similarly.
    for i in range(10):
        tokens = split_on_spaces_only(corpus)
        if len(tokens) <= valid_size:
            break
        random_index = random.randint(0, len(tokens) - valid_size)
        sub_valid_text = " ".join(tokens[random_index : random_index + valid_size])
        valid_data += (" " if valid_data else "") + sub_valid_text
        corpus = corpus.replace(sub_valid_text, "", 1)  # NEW: replace only first occurrence

    print("train_size: ", len(corpus))
    with open(os.path.join(DATA_DIR, "Shakespeare_clean_w_nl_train.txt"), "w") as text_file:
        text_file.write(corpus)

    print("valid_size: ", len(valid_data))
    with open(os.path.join(DATA_DIR, "Shakespeare_clean_w_nl_valid.txt"), "w") as text_file:
        text_file.write(valid_data)

    print("test_size: ", len(test_data))
    with open(os.path.join(DATA_DIR, "Shakespeare_clean_w_nl_test.txt"), "w") as text_file:
        text_file.write(test_data)
