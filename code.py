import random
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers


# ######################## PART 1: PROVIDED CODE ########################


def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.
    
    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        dd = data_dict

        if len(dd["premise"]) != len(dd["hypothesis"]) or len(dd["premise"]) != len(
                dd["label"]
        ):
            raise AttributeError("Incorrect length in data_dict")

    def __len__(self):
        return len(self.data_dict["premise"])

    def __getitem__(self, idx):
        dd = self.data_dict
        return dd["premise"][idx], dd["hypothesis"][idx], dd["label"][idx]


def train_distilbert(model, loader, device):
    model.train()
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in tqdm(loader):
        optimizer.zero_grad()

        inputs = model.tokenize(premise, hypothesis).to(device)
        target = target.to(device, dtype=torch.float32)

        pred = model(inputs)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_distilbert(model, loader, device):
    model.eval()

    targets = []
    preds = []

    for premise, hypothesis, target in loader:
        preds.append(model(model.tokenize(premise, hypothesis).to(device)))

        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


# ######################## PART 1: YOUR WORK STARTS HERE ########################
'''
1. Finetune DistilBERT for classification (40 pts)
In this part, you will use the NLI training data (same as A1) to finetune a DistilBERT model and
predict whether a premise entails a hypothesis or not. Just like the first assignment, you will
have to implement various parts of a custom nn.Module that loads a pretrained DistilBERT from
Huggingface transformers. You can learn more about DistilBERT here, but you can just assume
it’s a smaller version of BERT that remains fairly accurate.
'''


# You will have to implement the init function of the CustomDistilBert class. You will need to
# initialize the following attributes:
# ● self.distilbert
# ● self.tokenizer
# ● self.pred_layer
# ● self.sigmoid
# ● self.criterion
# For distilbert and tokenizer, you will need to use transformers, whereas pred_layer, sigmoid,
# and criterion require torch and correspond to questions you have previously answered in A1.


class CustomDistilBert(nn.Module):
    def __init__(self):
        """
        CustomDistilBert.__init__
        Note:
        ● Load the DistilBERT model's pretrained "base uncased" weights from the Huggingface
        repository. We want the bare encoder outputting hidden-states without any specific head
        on top.
        ● Load the corresponding pre-trained tokenizer using the same method.
        ● self.pred_layer takes the output of the model and predicts a single score (binary, 1 or 0),
        then pass the output to the sigmoid layer
        ● self.sigmoid should return torch's sigmoid activation.
        ● self.criterion should be the binary cross-entropy loss. You may use torch.nn here.
        """
        super().__init__()

        # TODO: your work below
        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.pred_layer = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    # vvvvv DO NOT CHANGE BELOW THIS LINE vvvvv
    def get_distilbert(self):
        return self.distilbert

    def get_tokenizer(self):
        return self.tokenizer

    def get_pred_layer(self):
        return self.pred_layer

    def get_sigmoid(self):
        return self.sigmoid

    def get_criterion(self):
        return self.criterion

    # ^^^^^ DO NOT CHANGE ABOVE THIS LINE ^^^^^

    def assign_optimizer(self, **kwargs):
        """
        CustomDistilBert.assign_optimizer
        This assigns the Adam optimizer to this model's parameters (self) and returns the optimizer.
        :param kwargs:
        :return:
        """
        # TODO: your work below
        return torch.optim.Adam(self.parameters(), **kwargs)

    def slice_cls_hidden_state(
            self, x
    ) -> torch.Tensor:
        """
        Edit the method CustomDistilBert.slice_cls_hidden_state. This is a helper method that will be
        used inside forward, and will convert the output of your transformer model to something that can
        be input in the prediction layer.
        CustomDistilBert.slice_cls_hidden_state
        Using the output of the model, return the last hidden state of the CLS token.
        ParameterTypeDescription
        xBaseModelOutputThe output of the distilbert model. You need to retrieve
        the hidden state of the last output layer, then slice it to
        obtain the hidden representation. The last hidden state
        has shape: [batch_size, sequence_length,
        hidden_size]
        ReturnsDescription
        Tensor[batch_size,
        hidden_size]The last layer's hidden state representing the [CLS] token.
        Usually, CLS is the first token in the sequence.
        :param x:
        :return:
        """
        # TODO: your work below
        return x.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

    def tokenize(
            self,
            premise: "list[str]",
            hypothesis: "list[str]",
            max_length: int = 128,
            truncation: bool = True,
            padding: bool = True,
    ):
        """
        Use the get_tokenizer function implemented in 2.1 to write the method
        CustomDistilBert.tokenize. That method is specifically to help you understand how the
        tokenizer works, and should be fairly straightforward.
        This function will be applied to the premise and hypothesis (list of str) to obtain the inputs for
        your model. You will need to use the Huggingface tokenizer returned by get_tokenizer().
        ParameterTypeDescription
        premiselist of strThe first text to be input in your model.
        hypothesislist of strThe second text to be input in your model.
        For the remaining params, see documentations.
        ReturnsDescription
        BatchEncodingA dictionary-like object that can be given to the model (you
        can find out how by reading the docs)
        :param premise:
        :param hypothesis:
        :param max_length:
        :param truncation:
        :param padding:
        :return:
        """
        # TODO: your work below
        return self.tokenizer(
            premise,
            hypothesis,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )

    def forward(self, inputs):
        """
        Given the output of your tokenizer (a BatchEncoding object), you will have to pass through your
        custom DistilBert model and output a score between 0 and 1 for each element in your batch;
        this score represents whether there’s an entailment or not.
        CustomDistilBert.forward
        Note: In the original BERT paper, the output representation of CLS is used for classification.
        You will need to slice the output of your DistilBERT to obtain the representation before giving it
        to the last layer with sigmoid activation.
        :param inputs:
        :return:
        """
        # TODO: your work below
        x = self.distilbert(**inputs, return_dict=True)
        x = self.slice_cls_hidden_state(x)
        x = self.pred_layer(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x


# ######################## PART 2: YOUR WORK HERE ########################
def freeze_params(model):
    """
    Before starting, you will need to freeze all the parameters (including the embedding!). This is
    because prompt tuning relies on tuning a very small number of fixed parameters (aka “prompts”,
    since they are inserted as input embeddings to the model). Thus, everything else, including the
    input word embeddings, are not trainable.
    :param model:
    :return:
    """
    # TODO: your work below
    for param in model.parameters():
        param.requires_grad = False


def pad_attention_mask(mask, p):
    """
    Pad the start of the sequence p times of the attention_mask (which is one of the various
    outputs of a Huggingface tokenizer) because the sequence length has changed. Find the
    correct value based on Huggingface documentations.
    :param mask:
    :param p:
    :return:
    """
    # TODO: your work below
    return F.pad(mask, (p, 0), value=1)


class SoftPrompting(nn.Module):
    def __init__(self, p: int, e: int):
        super().__init__()
        self.p = p
        self.e = e

        self.prompts = torch.randn((p, e), requires_grad=True)

    def forward(self, embedded):
        """
        This takes the output of model.embeddings and adds the soft prompts, as described in the
        paper. The prompts must be added at the start of the sequence.
        ParameterTypeDescription
        embeddedTensor[B, L, E]This corresponds to model.embeddings (where model is
        a Huggingface transformer)
        ● B: Batch size
        ● L: Sequence Length
        ● E: Embedding dimension (same as e)ReturnsDescription
        Tensor[B, L+p, E]The input_embed to be given to the model, but with the added
        :param embedded:
        :return:
        """
        # TODO: your work below
        p = self.prompts.unsqueeze(0).repeat(embedded.size(0), 1, 1).to(embedded.device)
        return torch.cat([p, embedded], dim=1)


# ######################## PART 3: YOUR WORK HERE ########################
"""
3. Open-domain question answering (45 pts)
In this part, you will need to implement a model similar to Dense Passage Retrieval for
Open-Domain Question Answering by Karpukhin et al (2020). Before starting, please go over
the paper, and optionally read this blog post section on open-domain QA and skim over the
DPR code repository.
For the assignment, you are given CSV files in data/qa that correspond to the training Q&A
pairs, validation pairs, test questions, and answers.csv that contains all the answers. The data is
taken from a Q&A forum about cooking, and at the end of this section you will have built a
model that can automatically retrieve (i.e. search) relevant answers for any question about
cooking.
"""


def load_models_and_tokenizer(q_name, a_name, t_name, device='cpu'):
    """
    In part 1, you used an object-oriented approach to load a custom distilbert model. This time, you
    need to load two models (one for encoding the questions, another for encoding the candidate
    answers, and they have separate weights but same embedding dimensions). You will use a
    simpler approach this time; simply build functions that load and return the models, perform
    slicing, and generate tokens, all separate from OOP.
    load_models_and_tokenizer
    For this, we will be testing with the string 'google/electra-small-discriminator' but
    your function should work with other names as well. You do not review ELECTRA, other than
    knowing that this is a pretrained model that performs well but is much smaller than DistilBERT
    and has smaller embedding size. You should be comfortable loading other models from
    Huggingface, as hundreds of them exist.
    ParameterTypeDescription
    q_namestrName of the model that will be passed to transformers to
    automatically load the pre-trained version, and used for encoding
    questions. It must be something that can be found on
    Huggingface Hub, just like you’ve previously done.
    a_namestrName of the model encoding the answers.
    t_namestrName of the tokenizer, which will be passed to transformers to
    automatically load the corresponding pre-trained tokenizer.
    devicestr“cuda” or “cpu”, the models will be loaded here.
    ReturnsType
    Descriptionq_enctransformers
    ModelThe question encoder model, which is a huggingface
    transformer. This corresponds to q_name
    a_enctransformers
    ModelThe candidate answers the encoder model, which is a
    huggingface transformer. This corresponds to a_name
    tokenizertransformers
    tokenizerThe tokenizer that will be used for each of the models. For
    simplicity, it will be shared, but in theory you could have
    separate tokenizers if you use different models for
    encoding questions and answers, but this is rare in
    practice.
    :param q_name:
    :param a_name:
    :param t_name:
    :param device:
    :return:
    """
    # TODO: your work below
    q_enc = transformers.ElectraModel.from_pretrained(q_name).to(device)
    a_enc = transformers.ElectraModel.from_pretrained(a_name).to(device)
    tokenizer = transformers.ElectraTokenizer.from_pretrained(t_name)
    return q_enc, a_enc, tokenizer


def tokenize_qa_batch(tokenizer, q_titles, q_bodies, answers, max_length=64):
    """
    tokenize_qa_batch
    Tokenize question titles/bodies and answers into two input batches. The following conditions
    apply:
    ● The input_ids returned should be Pytorch tensors
    ● The content should be truncated using the default method if it exceeds max_length
    ● You should only pad everything to the longest sequence in a batch
    ● The title and body should be tokenized together as a pair, separated with a [SEP] token.
    ● The answers should be tokenized separately
    ● q_titles,q_bodies, answers are all lists of the same length
    ParameterTypeDescription
    tokenizertransformers
    tokenizerA huggingface tokenizer returned by your previous
    function.
    q_titleslist of strThe list of titles of the questions
    q_bodieslist of strThe list of questions bodies (actual content)
    answerslist of strThe contents of the corresponding answers
    max_lengthintThe maximum length of the tokens, after which it is
    truncated
    ReturnsTypeDescription
    q_batchBatchEncodingA "BatchEncoding" (inherited from a dict) containing
    the question titles and bodies, which can be used as
    the input of a transformer model (which means the
    input IDs are PyTorch tensors)
    a_batch
    BatchEncoding
    A "BatchEncoding" (inherited from a dict) containing
    the answer bodies, which can be used as the input of a
    transformer model.
    :param tokenizer:
    :param q_titles:
    :param q_bodies:
    :param answers:
    :param max_length:
    :return:
    """
    # TODO: your work below.
    q_batch = tokenizer(
        q_titles,
        q_bodies,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    a_batch = tokenizer(
        answers,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return q_batch, a_batch


def get_class_output(model, batch):
    """
    Since this is similar to a previous question, it is left ungraded.
    ReturnsDescription
    BaseModelOutputThe output representation of the class token (for example [CLS])
    after encoding the tokenized text through your model.
    :param model:
    :param batch:
    :return:
    """
    # Since this is similar to a previous question, it is left ungraded
    # TODO: your work below.
    return model(**batch, return_dict=True).last_hidden_state[:, 0, :]


def inbatch_negative_sampling(Q: Tensor, P: Tensor, device: str = 'cpu') -> Tensor:
    """
    3.3 Batch: Implement in-batch negative sampling (7 pts)
    The in-batch negative sampling method uses the answers from other questions in the same
    batch as the negative examples (because they are unrelated as they are randomly taken from
    the training set). Please read the relevant section in the paper and implement it accordingly.
    inbatch_negative_sampling
    This function should take the tensors of questions Q and passages P, and use the in-batch
    negatives (as described in the paper) to compute a similarity matrix evaluated on each of the N
    questions with the M passages. Although we call it “sampling”, you are computing it over all
    passages in a batch rather than taking a subsample of the batch. You can find a way to do both
    the “negative sampling” and computing similarity at the same time; you can read the paper to
    find out how to do that.
    ParameterTypeDescription
    QTensor[N, E]The output representation of N question titles+bodies
    PTensor[M, E]The output representation of M answers (aka passages)
    devicestr“cuda” or “cpu”, the models will be loaded here.
    ReturnsDescription
    Tensor[N, M]The matrix of similarity score that results from in-batch negative
    sampling.
    :param Q:
    :param P:
    :param device:
    :return:
    """
    # TODO: your work below
    return torch.matmul(Q, P.T)


def contrastive_loss_criterion(S: Tensor, labels: Tensor = None, device: str = 'cpu'):
    """
    Contrastive loss is different from the loss functions we have previously been exposed to. You
    have previously seen loss functions for classification and for text generation, but retrieval
    requires something different. You will need to implement the loss function described in the DPR
    paper (please read the equations carefully), and the official source code shows how to
    implement it in Pytorch.
    contrastive_loss_criterion
    ParameterTypeDescription
    STensor[N, M]The matrix of similarity score that results from in-batch
    negative sampling.
    labelsTensor[N]The optional label indices between 0-M. For example [0, 2,
    M,..., 1] respectively indicate that the
    ● Passage #0 is the answer for Question #0
    ● Passage #2 is the answer for Question #1
    ● Passage #M is the answer for Question #2
    ● …
    ● Passage #1 is the answer for Question #N
    If labels=None, simply return a tensor with values such at
    Passage #0 corresponds to Question #0, P1 with Q1, etc.
    devicestr“cuda” or “cpu”, the models will be loaded here.
    ReturnsDescription
    Tensor(1)A scalar tensor on which you can call backward to initiate the backprop
    process. The value represents the loss, which means a lower value
    means the model has a low error when matching the questions with the
    passages.
    :param S:
    :param labels:
    :param device:
    :return:
    """
    # TODO: your work below
    if labels is None:
        labels = torch.arange(S.size(0), device=device)
    return torch.nn.functional.cross_entropy(S, labels)


def get_topk_indices(Q, P, k: int = None):
    """
    get_topk_indices
    Compute the dot-product similarity score (without normalizing or cosine scaling) and return the
    indices and scores for the top-k candidate answers for each question in Q.
    ParameterTypeDescription
    QTensor[N, E]The output class representation for question title+body
    PTensor[M, E]The output class representation for answer (aka
    passage)
    kintThe number of indices to return based on the similarity.When k=None, simply return everything in the sorted
    order.
    ReturnsTypeDescription
    indicesTensor[N, k] of intThe sorted indices of the most similar answers for each
    of N questions (from largest to smallest magnitude)
    scoresTensor[N, k] of intThe dot-product similarity score for the corresponding
    indices
    :param Q:
    :param P:
    :param k:
    :return:
    """
    # TODO: your work below
    scores = torch.matmul(Q, P.T)
    indices = torch.argsort(scores, dim=1, descending=True)
    if k is not None:
        indices = indices[:, :k]
        scores = scores.gather(1, indices)
    return indices, scores


def select_by_indices(indices: Tensor, passages: 'list[str]') -> 'list[str]':
    """
        select_by_indices
    ParameterTypeDescription
    indicesTensor[N, k] of intThe sorted indices of the most similar answers for each
    of N questions (from largest to smallest magnitude)
    passageslist of strA list of answers in the original textual format (before
    tokenization). The length should be greater than the
    largest index in indices (i.e. M).
    Returns
    Description
    list of list of str
    ●
    ●
    The outer lists correspond to answers for the the N questions
    The inner lists contain the k sorted answers in original text format
    :param indices:
    :param passages:
    :return:
    """
    # TODO: your work below
    return [[passages[i] for i in idx] for idx in indices]


def embed_passages(passages: 'list[str]', model, tokenizer, device='cpu', max_length=512):
    """
    First, set the model into the evaluation mode and disable gradients, then embed a list of
    passages.
    ParameterTypeDescription
    passageslist of strA list of answers in the original textual format (before
    tokenization). The length should be greater than the largest index
    in indices (i.e. M).
    modelThe Huggingface transformer model used to encode the text
    tokenizerThe Huggingface tokenizer used to encode the passages
    Returns
    Description
    Tensor[M, E]
    The output class representation for answer (aka passage)
    :param passages:
    :param model:
    :param tokenizer:
    :param device:
    :param max_length:
    :return:
    """
    # TODO: your work below
    model.eval()
    with torch.no_grad():
        encoded_passages = tokenizer(passages, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        outputs = model(**encoded_passages, return_dict=True)
        passage_embeddings = outputs.last_hidden_state[:, 0, :]
    return passage_embeddings


def embed_questions(titles, bodies, model, tokenizer, device='cpu', max_length=512):
    """
    First, set the model into the evaluation mode and disable gradients. Then, embed titles and
    bodies at the same time using the question model (review BERT to see how this is possible).
    ParameterTypeDescription
    titleslist of strA list of titles in the original textual format (before tokenization).
    The length should be greater than the largest index in indices (i.e.
    M).
    bodieslist of strThe bodies corresponding to each title.
    modelThe Huggingface transformer model used to encode the text
    tokenizerThe Huggingface tokenizer used to encode the passages
    ReturnsDescription
    Tensor[N, E]The output class representation for questions (titles with bodies)
    :param titles:
    :param bodies:
    :param model:
    :param tokenizer:
    :param device:
    :param max_length:
    :return:
    """
    # TODO: your work below
    model.eval()
    with torch.no_grad():
        # Combine titles and bodies with the [SEP] token
        combined_texts = ["[CLS] " + title + " [SEP] " + body for title, body in zip(titles, bodies)]
        encoded_questions = tokenizer(combined_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        outputs = model(**encoded_questions, return_dict=True)
        question_embeddings = outputs.last_hidden_state[:, 0, :]  # Use the pooled output for question embeddings
    return question_embeddings


def recall_at_k(retrieved_indices: 'list[list[int]]', true_indices: 'list[int]', k: int):
    """
    You can review the recall score here. Note that since we are returning k elements, we want to
    know if the correct answer is in one of the k elements, hence we are calling this the “recall”
    score, though it might feel different from how we use recall for binary classification.
    ParameterTypeDescription
    retrieved_indiceslist of list of
    intouter list: The retrieved results for each of N questions
    inner list: The k indices ranked in order of estimated
    relevance
    true_indiceslist of intThe correct index for each of N questions, same length as
    retrieved_indices’s outer list
    kintThe number of inner items in retrieved_indices to consider;
    k must be smaller than the length of the inner lists
    ReturnsDescriptionfloat
    A single score representing the recall at k score
    :param retrieved_indices:
    :param true_indices:
    :param k:
    :return:
    """
    # TODO: your work below
    return sum([1 if true in retrieved[:k] else 0 for retrieved, true in zip(retrieved_indices, true_indices)]) / len(retrieved_indices)


def mean_reciprocal_rank(retrieved_indices: 'list[list[int]]', true_indices: 'list[int]'):
    """
    Please refer to recall_at_k, except you are now returning the Mean Reciprocal Rank, and
    there’s no parameter k. You can read more about this on Wikipedia. The mean reciprocal rank is a statistic measure
    for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability
    of correctness. The reciprocal rank of a query response is the multiplicative inverse of the rank of the first
    correct answer: 1 for first place, 1⁄2 for second place, 1⁄3 for third place and so on. The mean reciprocal rank is
    the average of the reciprocal ranks of results for a sample of queries Q
    :param retrieved_indices:
    :param true_indices:
    :return:
    """
    # TODO: your work below
    assert len(retrieved_indices) == len(true_indices), "Length of retrieved_indices and true_indices must be equal."

    reciprocal_ranks = []
    for retrieved, true in zip(retrieved_indices, true_indices):
        if true in retrieved:
            rank = retrieved.index(true) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ######################## PART 4: YOUR WORK HERE ########################


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import f1_score  # Make sure sklearn is installed

    random.seed(2022)
    torch.manual_seed(2022)

    # Parameters (you can change them)
    sample_size = 2500  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 2
    num_words = 50000

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 1: TEST CODE ######################
    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data/nli")
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = (
            train_raw["premise"]
            + train_raw["hypothesis"]
            + valid_raw["premise"]
            + valid_raw["hypothesis"]
    )

    print("=" * 80)
    print("Running test code for part 1")
    print("-" * 80)

    train_loader = torch.utils.data.DataLoader(
        NLIDataset(train_raw), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        NLIDataset(valid_raw), batch_size=batch_size, shuffle=False
    )

    model = CustomDistilBert().to(device)
    optimizer = model.assign_optimizer(lr=1e-4)

    for epoch in range(n_epochs):
        loss = train_distilbert(model, train_loader, device=device)

        preds, targets = eval_distilbert(model, valid_loader, device=device)
        preds = preds.round()

        score = f1_score(targets.cpu(), preds.cpu())
        print("Epoch:", epoch)
        print("Training loss:", loss)
        print("Validation F1 score:", score)
        print()

    # ###################### PART 2: TEST CODE ######################
    freeze_params(model.get_distilbert())  # Now, model should have no trainable parameters

    sp = SoftPrompting(p=5, e=model.get_distilbert().embeddings.word_embeddings.embedding_dim).to(device)
    batch = model.tokenize(
        ["This is a premise.", "This is another premise."],
        ["This is a hypothesis.", "This is another hypothesis."],
    ).to(device)
    batch.input_embedded = sp(model.get_distilbert().embeddings(batch.input_ids))
    batch.attention_mask = pad_attention_mask(batch.attention_mask, 5)

    # ###################### PART 3: TEST CODE ######################
    # Preliminary
    bsize = 8
    qa_data = dict(
        train=pd.read_csv('data/qa/train.csv'),
        valid=pd.read_csv('data/qa/validation.csv'),
        answers=pd.read_csv('data/qa/answers.csv'),
    )

    q_titles = qa_data['train'].loc[:bsize - 1, 'QuestionTitle'].tolist()
    q_bodies = qa_data['train'].loc[:bsize - 1, 'QuestionBody'].tolist()
    answers = qa_data['train'].loc[:bsize - 1, 'Answer'].tolist()

    # Loading huggingface models and tokenizers
    name = 'google/electra-small-discriminator'
    q_enc, a_enc, tokenizer = load_models_and_tokenizer(q_name=name, a_name=name, t_name=name)

    # Tokenize batch and get class output
    q_batch, a_batch = tokenize_qa_batch(tokenizer, q_titles, q_bodies, answers)

    q_out = get_class_output(q_enc, q_batch)
    a_out = get_class_output(a_enc, a_batch)

    # Implement in-batch negative sampling
    S = inbatch_negative_sampling(q_out, a_out)

    # Implement contrastive loss
    loss = contrastive_loss_criterion(S)
    # or
    # > loss = contrastive_loss_criterion(S, labels=...)

    # Implement functions to run retrieval on list of passages
    titles = q_titles
    bodies = q_bodies
    passages = answers + answers
    Q = embed_questions(titles, bodies, model=q_enc, tokenizer=tokenizer, max_length=16)
    P = embed_passages(passages, model=a_enc, tokenizer=tokenizer, max_length=16)

    indices, scores = get_topk_indices(Q, P, k=5)
    selected = select_by_indices(indices, passages)

    # Implement evaluation metrics
    retrieved_indices = [[1, 2, 12, 4], [30, 11, 14, 2], [16, 22, 3, 5]]
    true_indices = [1, 2, 3]

    print("Recall@k:", recall_at_k(retrieved_indices, true_indices, k=3))

    print("MRR:", mean_reciprocal_rank(retrieved_indices, true_indices))
