from torch.utils.data import Dataset
from nltk.corpus import stopwords
import collections
import numpy as np
import torchtext
import warnings
import shutil
import torch
import gdown
import copy
import nltk
import json
import re
import os
nltk.download('stopwords')

class Token:
  def __init__(self, default_format, properties):
    if not callable(default_format):
      self.default_format = lambda: default_format
    else:
      self.default_format = default_format
    self.properties = {**properties}

  def __getitem__(self, ix):
    return self.properties.__getitem__(ix)

  def __setitem__(self, ix, val):
    return self.properties.__setitem__(ix, val)

  def to_string(self, format_):
    return format_.format(**self.properties)

  def __string_rep(self):
    return self.to_string(self.default_format())

  def __str__(self):
    return self.__string_rep()

  def __repr__(self):
    return self.__str__() + ": " + self.properties.__repr__()

  def __hash__(self):
    return self.__string_rep().__hash__()

  def __eq__(self, other):
    return self.__string_rep() == other.__string_rep()

  def __gt__(self, other):
    return self.__string_rep() > other.__string_rep()

  def __lt__(self, other):
    return self.__string_rep() < other.__string_rep()


class Sentence:
  def __init__(self, json_, fmt, get_ancestor):
    self.json = {**json_}
    self.fmt = fmt
    # Lowercase lemmata (plural of lemma)
    for t in self.json['tokens']:
      t['lemma'] = t['lemma'].lower()
    # Create tokens
    self.tokens = \
    [
      Token(self.getFormat, token)
      for token in self.json['tokens']
    ]
    # Find each token's ancestor
    if get_ancestor:
      tree = self.makeParseTree()
      for ix, token in enumerate(self.tokens):
        tree_ix = tree.leaf_treeposition(ix)
        token["ancestor"] = tree[tree_ix[:-2]].label()

  def __getitem__(self, ix):
    return self.tokens[ix]

  def __setitem__(self, ix, val):
    self.tokens[ix] = val

  def getFormat(self):
    return self.fmt

  def setFormat(self, newFmt):
    self.fmt = newFmt

  def withoutHTML(self):
    # Remove __HTML__ tokens used in swapping to highlight the swapped phrases
    without = {**self.json}
    # Remove from parse
    without['parse'] = re.sub('\(__HTML__[^)]+\)', '', self.json["parse"])
    # Remove from token list
    without['tokens'] = [s for s in self.json['tokens'] if s['pos'] != "__HTML__"]
    # Done
    return Sentence(without, self.getFormat(), False)

  def withoutPunctuation(self):
    without = Sentence({**self.json}, self.getFormat(), False)
    # Remove punctuation from parse
    parse = self.json["parse"]
    punct_ix = list(re.finditer("\([^a-zA-Z0-9(]\S* \S*[^a-zA-Z0-9)]\)", parse))
    for m in reversed(punct_ix):
      parse = parse[:m.start()]+ parse[m.end():]
    without.json["parse"] = parse
    # Remove punctuation from tokens
    tokens = []
    for ix, t in enumerate(self.tokens):
      if t["pos"][0].isalpha():
        tokens.append(Token(without.getFormat, {**t.properties}))
      elif 0 < len(tokens):
        tokens[-1]["after"] += t["after"]
    without.tokens = tokens
    # Done, leave the other fields untouched
    return without

  def withoutStopWords(self):
    without = Sentence(self.json, self.getFormat(), False)
    # Remove stopwords from tokens
    tokens = []
    for ix, t in enumerate(self.tokens):
      if t["lemma"].lower() not in stopwords.words('english'):
        tokens.append(Token(without.getFormat, {**t.properties}))
    without.tokens = tokens
    # Done, leave the other fields untouched
    return without

  def makeParseTree(self):
    from nltk.tree import Tree
    parse = self.json['parse']
    # Replace words with indices in parse string
    indices = list(enumerate(re.finditer('\s[^ )]+\)', parse)))
    for ix, match in reversed(indices):
      parse = parse[:match.start() + 1] + str(ix) + parse[match.end()-1:]
    # Use parse string to create a tree
    tree = Tree.fromstring(parse)
    # Replace indices in tree with tokens
    for lix, leaf in enumerate([leaf for leaf in tree.leaves() if leaf.isnumeric()]):
      tree[tree.leaf_treeposition(lix)] = self.tokens[int(leaf)]
    return tree

  def to_string(self, fmt=None, after=None):
    if fmt is None:
      fmt = self.fmt
    out = ''
    for t in self.tokens:
      out += t.to_string(fmt) + (t["after"] if after is None else after)
    return out

  def __str__(self):
    return self.to_string()

  def __repr__(self):
    return self.__str__().strip() + ": " + repr(self.json)

  def __len__(self):
    return len(self.tokens)

  @staticmethod
  def __findLabelInTree(tree, label):
    return [p for p in tree.treepositions() if isinstance(tree[p], nltk.tree.Tree) and label == tree[p].label()]

  @staticmethod
  def __getParseStringFromTreeWithTokens(tree):
    tree_copy = tree.copy(deep=True)
    for lix in range(len(tree_copy.leaves())):
      tree_copy[tree_copy.leaf_treeposition(lix)] = \
        tree_copy[tree_copy.leaf_treeposition(lix)].to_string("{originalText}")
    return str(tree_copy)

  @staticmethod
  def __treeToSentence(tree, fmt):
    return Sentence(
      {
        "parse": Sentence.__getParseStringFromTreeWithTokens(tree),
        "tokens": [token.properties for token in tree.leaves()]
      },
      fmt, False
    )

  @staticmethod

  def SwapPhrases(sentence_1, sentence_2, label, validate_phrase=None, prefix=None, suffix=None):
    from nltk.tree import Tree
    import numpy as np

    # Build a tree from each sentence
    tree_1 = sentence_1.withoutHTML().makeParseTree()
    tree_2 = sentence_2.withoutHTML().makeParseTree()
    # Find phrase in both trees
    tree_1_phrases = Sentence.__findLabelInTree(tree_1, label)
    tree_2_phrases = Sentence.__findLabelInTree(tree_2, label)
    # If there is nothing to swap, return
    if 0 == len(tree_1_phrases) or 0 == len(tree_2_phrases):
      return (None, None, None, None)
    # Pick a phrase at random from each sentence
    tree_1_index = tree_1_phrases[np.random.randint(len(tree_1_phrases))]
    tree_2_index = tree_2_phrases[np.random.randint(len(tree_2_phrases))]
    # Validate the phrases to swap
    if validate_phrase is not None \
      and not validate_phrase(tree_1, tree_1_index, tree_2, tree_2_index):
        return (None, None, None, None)
    # DEBUG #
    if prefix is not None:
      # We want prefix to be a list with a prefix for both sentences
      if isinstance(prefix, str):
        prefix = [prefix, prefix]
      # Insert a new HTML token before the beginning of the phrase
      tree_1[tree_1_index].insert(
        0, Tree(
          "__HTML__", [Token(
            sentence_1.fmt,
            {
              'originalText': prefix[0],
              'word': prefix[0],
              'lemma': prefix[0],
              'pos': "__HTML__",
              'before': "",
              'after': "",
              'index': None,
              'characterOffsetBegin': None,
              'characterOffsetEnd': None
            }
          )]
        )
      )
      tree_2[tree_2_index].insert(
        0, Tree(
          "__HTML__", [Token(
            sentence_2.fmt,
            {
              'originalText': prefix[1],
              'word': prefix[1],
              'lemma': prefix[1],
              'pos': "__HTML__",
              'before': "",
              'after': "",
              'index': None,
              'characterOffsetBegin': None,
              'characterOffsetEnd': None
            }
          )]
        )
      )
    if suffix is not None:
      if isinstance(suffix, str):
        suffix = [suffix, suffix]
      # Append a new HTML token to the phrase
      tree_1[tree_1_index].append(
        Tree(
          "__HTML__", [Token(
            sentence_1.fmt,
            {
              'originalText': suffix[0],
              'word': suffix[0],
              'lemma': suffix[0],
              'pos': "__HTML__",
              'before': "",
              'after': "",
              'index': None,
              'characterOffsetBegin': None,
              'characterOffsetEnd': None
            }
          )]
        )
      )
      tree_2[tree_2_index].append(
        Tree(
          "__HTML__", [Token(
            sentence_2.fmt,
            {
              'originalText': suffix[1],
              'word': suffix[1],
              'lemma': suffix[1],
              'pos': "__HTML__",
              'before': "",
              'after': "",
              'index': None,
              'characterOffsetBegin': None,
              'characterOffsetEnd': None
            }
          )]
        )
      )
    #########
    # Save modified original sentences
    mod_sent_1 = Sentence.__treeToSentence(tree_1, sentence_1.fmt)
    mod_sent_2 = Sentence.__treeToSentence(tree_2, sentence_2.fmt)
    # Swap
    swap = tree_1[tree_1_index]
    tree_1[tree_1_index] = tree_2[tree_2_index]
    tree_2[tree_2_index] = swap
    swap_sent_1 = Sentence.__treeToSentence(tree_1, sentence_1.fmt)
    swap_sent_2 = Sentence.__treeToSentence(tree_2, sentence_2.fmt)
    return \
    (
      swap_sent_1, swap_sent_2, mod_sent_1, mod_sent_2
    )

archive = "in/Automating-Intention-Mining-parsed-data.tar.gz"
url = "https://drive.google.com/uc?id=1MYR04EN9wyEw5C-RhpAmX5Xnat-jiBSy"
print("Downloading {}: ".format(archive), end="")
if not os.path.isfile(archive):
  gdown.download(url, archive, 0)
else:
  print('file already exists. Skipping download.')
print("done")

# Remove old paths
parsed_folder = 'parsed'
if os.path.exists(parsed_folder):
  shutil.rmtree(parsed_folder)

print("Extracting files... ")
shutil.unpack_archive(archive)
print("Extracting files... done")

# Load data
projects = ['DECA', 'bootstrap', 'docker', 'tensorflow', 'vscode']
categories = [
  'aspect evaluation', 'feature request', 'information giving',
  'information seeking', 'problem discovery', 'solution proposal', 'others'
]

_parsed_cat_proj = {}
for c in categories:
  _parsed_cat_proj[c] = {}
  for p in projects:
    with open(os.path.join(parsed_folder, p, c + ".json"), 'r', encoding='latin-1') \
      as f:
      j = json.load(f)
      assert c == j["docId"]
      _parsed_cat_proj[c][p] = j["sentences"]
      # for s in j["sentences"]:
      #   _parsed_cat_proj[c][p].append(Sentence(s, "{lemma}/{pos}"))

def GetAllText(
  word="word", show_pos=False, remove_punctuation=False, remove_stopwords=False,
  get_ancestors=True, projects_to_exclude=None
):

  if word == "word":
    fmt = "{originalText}"
  elif word == "lemma":
    fmt = "{lemma}"
  else:
    raise Exception("Value (\"{}\") for @word not recognized.")
  if show_pos:
    fmt += "/{pos}"

  if remove_stopwords:
    constructor1 = lambda *args: Sentence(*args).withoutStopWords()
  else:
    constructor1 = lambda *args: Sentence(*args)

  if remove_punctuation:
    constructor2 = lambda *args: constructor1(*args).withoutPunctuation()
  else:
    constructor2 = constructor1

  if projects_to_exclude is None:
    projects_to_exclude = []
  elif isinstance(projects_to_exclude, str):
    projects_to_exclude = [projects_to_exclude]

  projects_to_exclude = [p.lower() for p in projects_to_exclude]

  return \
  [
    constructor2(sentence, fmt, get_ancestors)
    for category_name, projects in _parsed_cat_proj.items()
    for project_name, project_text in projects.items()
    for sentence in project_text
    if project_name.lower() not in projects_to_exclude
  ]


def GetTextByCategories(
  word="word", show_pos=False, remove_punctuation=False, remove_stopwords=False,
  get_ancestors=True, projects_to_exclude=None
):

  if word == "word":
    fmt = "{originalText}"
  elif word == "lemma":
    fmt = "{lemma}"
  else:
    raise Exception("Value (\"{}\") for @word not recognized.")
  if show_pos:
    fmt += "/{pos}"

  if remove_stopwords:
    constructor1 = lambda *args: Sentence(*args).withoutStopWords()
  else:
    constructor1 = lambda *args: Sentence(*args)

  if remove_punctuation:
    constructor2 = lambda *args: constructor1(*args).withoutPunctuation()
  else:
    constructor2 = constructor1

  if projects_to_exclude is None:
    projects_to_exclude = []
  elif isinstance(projects_to_exclude, str):
    projects_to_exclude = [projects_to_exclude]

  projects_to_exclude = [p.lower() for p in projects_to_exclude]

  return \
  {
    category_name:
    [
      constructor2(sentence, fmt, get_ancestors)
      for project_name, project_text in projects.items()
      for sentence in project_text
      if project_name.lower() not in projects_to_exclude
    ]
    for category_name, projects in _parsed_cat_proj.items()
  }

# @title StanfordFiles(Dataset) {display-mode: "form"}

class StanfordFiles(Dataset):

    def __init__(self, projects):
      self.text_vocab = None
      self.label_vocab = None
      self.vec = None

      if projects is None:
        projects = []

      all_projects = ["DECA", "bootstrap", "docker", "tensorflow", "vscode"]
      projects_to_exclude = [p for p in all_projects if p not in projects]
      text_by_cat = GetTextByCategories(
        word="word", show_pos=False, get_ancestors=True,
        remove_punctuation=False, remove_stopwords=False,
        projects_to_exclude=projects_to_exclude
      )

      self.text = []
      self.processed = []
      self.labels = []
      for cat, sentences in text_by_cat.items():
        for sent in sentences:
          self.text.append(sent)
          self.labels.append(cat)
          self.processed.append(Sentence(sent.json, sent.getFormat(), False))

    @staticmethod
    def Preprocess(dataset):
      dataset.processed = []
      for sent in dataset.text:
        dataset.processed.append(sent.withoutHTML().withoutStopWords().withoutPunctuation())
        dataset.processed[-1].setFormat("{lemma}")

    def build_vocabulary(self):
        # Warning: assumes that text has already been preprocessed and split!
        # Create vocabularies from text and label
        self.text_vocab = torchtext.vocab.Vocab(
            collections.Counter(
                [word for (sentence, label) in self for word in sentence]
            ), specials=['<unk>', '<pad>'], specials_first=True
        )
        self.label_vocab = torchtext.vocab.Vocab(
            collections.Counter([label for (sentence, label) in self]) , specials=[]
        )

    def word2tensor(self, pad_length, dataset=None):
        # Transforms text and labels to numerical indices using vocabulary built
        # using this dataset.
        #
        #   pad_length      Length to pad to (e.g., pad_length = 100, but
        #                   sentence is 75 characters, then 25 <pad> characters
        #                   will be added.
        #   dataset         Dataset to which to apply this. Iterating dataset
        #                   should return a (text, label) tuple. Default: self
        #                   (if dataset=None, use self).
        #
        # Warning: assumes that text has already been preprocessed and split!
        if dataset is None:
            dataset = self
        if isinstance(dataset, StanfordFiles):
            vec = [[]] * len(dataset.processed)
            lab = [[]] * len(dataset.labels)
        # Text to numerical indices (tensors)
        for ix in range(len(dataset)):
            sentence = [self.text_vocab.stoi[word] for word in dataset[ix][0]]
            label = self.label_vocab.stoi[dataset[ix][1]]

            if pad_length is not None and len(sentence) > pad_length:
                warnings.warn(
                    'The following sentence has {} characters which is longer '\
                    'than your padding length ({}).\nSentence = "{}"'\
                    .format(len(sentence), pad_length, sentence)
                )
            elif pad_length is not None:
                sentence = sentence + [self.text_vocab.stoi['<pad>']]*(pad_length-len(sentence))

            if isinstance(dataset, StanfordFiles):
                vec[ix] = torch.tensor(sentence)
                lab[ix] = torch.tensor(label)
            else:
                dataset[ix] = (torch.tensor(sentence), torch.tensor(label))
        if isinstance(dataset, StanfordFiles):
            dataset.vec = vec
            dataset.labels = lab

    def clone(self):
        # Create a copy of this dataset
        cloned = StanfordFiles(projects=None)
        cloned.text = [*self.text]
        cloned.processed = [*self.processed]
        cloned.labels = [*self.labels]
        return cloned
        
    def shuffle(self):
        # Order the data such that the classes are ordered and proportionally
        # represented (e.g., if for each element from categories 1 and 2 there
        # are two from category 3, then [1, 2, 3, 3, 1, 2, 3, 3, ...])
        # but such that the instances of within the categories are random.
        old_text = self.text
        old_processed = self.processed
        old_vec = self.vec
        old_labels = self.labels
        L = np.array([{lab: ix for ix, lab in enumerate(set(self.labels))}.get(lab) for lab in self.labels])

        index = [np.where(L == ii)[0] for ii in set(L)]
        permuted = [np.random.permutation(ii.size) for ii in index]

        # CATegory ID's Sorted by category Size (number of elements in the category)
        cat_id_ss = np.argsort([c.size for c in index])

        ix = 0;
        ct = [0]*len(index)
        self.text = [[]]*len(self.text)
        self.processed = [[]]*len(self.processed)
        if self.vec is not None:
            self.vec = [[]]*len(self.vec)
        self.labels = [[]]*len(self.labels)
        for ii in range(len(L)):
            # Current category
            cix = cat_id_ss[ix]
            # Next category
            nix = cat_id_ss[(1 + ix) % len(cat_id_ss)]
            # Finished with a category? Remove it from the list of ID's
            while ct[cix] >= len(permuted[cix]):
                # remove category
                cat_id_ss = np.delete(cat_id_ss, ix)
                # recalculate ix
                ix = ix % len(cat_id_ss)
                # recalculate current and next category
                cix = cat_id_ss[ix]
                nix = cat_id_ss[(1 + ix) % len(cat_id_ss)]

            # Array index (within array index [see below])
            ax = permuted[cix][ct[cix]]
            # Permuted array index
            px = index[cix][ax]

            # Label = OldLabel[permuted_array_index]
            self.labels[ii] = old_labels[px]
            # Text = OldText[permuted_array_index]
            self.text[ii] = old_text[px]
            self.processed[ii] = old_processed[px]
            if self.vec is not None:
                self.vec[ii] = old_vec[px]
            # Count this as one more instance of this category
            ct[cix] = 1 + ct[cix]
            # How do the category proportions compare? If the category has a higher proportion
            # of elements in the resulting array, move on to the next category.
            if ct[cix] / len(index[cix]) >= ct[nix] / len(index[nix]) :
                ix = (1 + ix) % len(cat_id_ss)


    def sort_by_labels(self):
        # Sort dataset according to label
        to_sort_by_labels = [self.labels, self.text]
        if self.processed is not None:
            to_sort_by_labels.append(self.processed)
        if self.vec is not None:
            to_sort_by_labels.append(self.vec)

        sorted_by_labels = sorted(zip(*to_sort_by_labels), key=lambda x: x[0])

        self.labels = []
        self.text = []
        if self.processed is not None:
          self.processed = []
        if self.vec is not None:
          self.vec = []

        for ltpv in sorted_by_labels:
          self.labels.append(ltpv[0])
          self.text.append(ltpv[1])
          if len(ltpv) > 2:
            self.processed.append(ltpv[2])
          if len(ltpv) > 3:
            self.vec.append(ltpv[3])


    def split(self, training_validation_split):
        idx = int(np.ceil(len(self) * training_validation_split))

        train = StanfordFiles(projects=None)
        train.text = self.text[:idx]
        train.processed = self.processed[:idx]
        train.labels = self.labels[:idx]

        validation = StanfordFiles(projects=None)
        validation.text = self.text[idx:]
        validation.processed = self.processed[idx:]
        validation.labels = self.labels[idx:]

        if self.vec is not None:
            train.vec = self.vec[:idx]
            validation.vec = self.vec[idx:]

        return train, validation


    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        if 0 > idx or idx >= len(self):
            raise IndexError

        if self.vec is not None:
            t = self.vec[idx]
        else:
            t = self.processed[idx]

        if self.labels is None:
            label = idx
        else:
            label = self.labels[idx]

        return (t, label)


    def __setitem__(self, idx, value):
        if not isinstance(value, tuple):
            raise TypeError

        if isinstance(value[0], torch.Tensor) and self.vec is not None:
            self.vec[idx] = value[0];
        else:
            self.processed[idx] = value[0];

        if self.labels is not None:
            self.labels[idx] = value[1]

print(
  '\n\n====================================================================\n'
  + 'functions:\n'
  + '  def GetAllText(\n'
  + '    word, show_pos,\n'
  + '    remove_punctuation, remove_stopwords,\n'
  + '    get_ancestors, projects_to_exclude\n'
  + '  )\n'
  + '    Returns all text as a list.\n'
  + '    (See parameter descriptions below)\n'
  + '\n'
  + '  def GetTextByCategories(\n'
  + '    word, show_pos,\n'
  + '    remove_punctuation, remove_stopwords,\n'
  + '    get_ancestors, projects_to_exclude\n'
  + ')\n'
  + '    Returns a dictionary with category names as keys and lists of\n'
  + '    sentences belonging to that category as values.\n'
  + '\n'
  + '    Parameters:\n'
  + '      word: "word" (original text) or lemma.\n'
  + '      show_pos: show Part-Of-Speech tag?\n'
  + '      remove_punctuation: self-explanatory.\n'
  + '      remove_stopwords: self-explanatory.\n'
  + '      get_ancestors: if True, get each token\'s ancestor\'s label (chunk).\n'
  + '      projects_to_exclude: self-explanatory. E.g., ["DECA", "vscode"]\n'
  + '\n'
  + '\n'
  + 'classes:\n'
  + '  class Token\n'
  + '    Token(default_format, properties)\n'
  + '      default_format: function returning format string (e.g.,\n'
  + '        "{{lemma}}")\n'
  + '      properties: json token dictionary\n'
  + '    Token[property]\n'
  + '      returns property\n'
  + '    str(Token)\n'
  + '      prints token using default_format\n'
  + '    Token.to_string(format)\n'
  + '      prints token using @format\n'
  + '\n'
  + '  class Sentence\n'
  + '    Sentence(json, format, get_ancestor)\n'
  + '      json: json output produced by Stanford parser.\n'
  + '      format: format used to display sentence\'s tokens (e.g.,\n'
  + '        "{{originalText}}/{{pos}}" to get "word/NN")\n'
  + '      get_ancestor: boolean; if true, get ancestor label (chunk).\n'
  + '    Sentence[ix]\n'
  + '      get/set token using []\n'
  + '    Sentence.makeParseTree()\n'
  + '      create parse tree representation of sentence\n'
  + '    Sentence.setFormat(newFmt)\n'
  + '      change sentence format\n'
  + '    Sentence.withoutPunctuation()\n'
  + '      remove punctuation from Sentence\n'
  + '    Sentence.withoutStopWords()\n'
  + '      remove stop words from Sentence\n'
  + '    Sentence.to_string(fmt, after)\n'
  + '      transform sentence to string using @format and printing @after\n'
  + '      between tokens. Set @after=None to print the token with the text\n'
  + '      that originally followed it.\n'
  + '    str(Sentence)\n'
  + '      transform sentence to string using current format\n'
  + '\n'
  + '  class StanfordFiles\n'
  + '    StanfordFiles(projects)\n'
  + '      create a dataset containing the sentences from the projects\n'
  + '      in @projects (e.g., StanfordFiles(["docker", "tensorlow"]))\n'
  + '    static StanfordFiles.Preprocess(dataset)\n'
  + '      preprocess @dataset (remove stop words, remove punctuation,\n'
  + '      and get lemma).\n'
  + '    StanfordFiles.build_vocabulary()\n'
  + '      create vocabulary (word->int and int->word) for text and labels.\n'
  + '    StanfordFiles.word2tensor(pad_length, [dataset])\n'
  + '      convert words to embedding indices using vocabulary and pad\n'
  + '      sentences to @pad_length. Words can come from \$self or from\n'
  + '      @dataset.\n'
  + '    StanfordFiles.shuffle()\n'
  + '      stratify and shuffle data. Often used before split().\n'
  + '      WARNING: data will then be in a label_1, label_2, label_2, ...\n'
  + '      If you do not want this, make sure to set shuffle=True when\n'
  + '      creating the DataLoader\n'
  + '    StanfordFiles.split(training_validation_split)\n'
  + '      split dataset into training and validation sets. The training set\n'
  + '      contains @training_validation_split portion of all sentences and\n'
  + '      the validation set, 1 - @training_validation_split.\n'
  + '    StanfordFiles[ix]\n'
  + '      You can use [] to get/set a sentence/label pair.\n'
  + '    len(StanfordFiles)\n'
  + '      You can use len(dataset) to get the number of elements in the\n'
  + '      dataset.\n'
  + '\n'
  + '======================================================================\n'
)
