import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP Conj NP VP | NP VP | NP VP Conj VP | NP Conj NP VP
AP -> Adj | AP Adj
NP -> N | Det N | Det AP N | P NP | NP P NP
AdvP -> Adv | AdvP Adv
VP -> V | VP NP | AdvP VP | VP AdvP | VP NP AdvP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokenedSentence = nltk.word_tokenize(sentence)

    preProcessedSentence = []

    for word in tokenedSentence:
        if hasAlpha(word):
            preProcessedSentence.append(word.lower())

    return preProcessedSentence


def hasAlpha(word):
    for letter in word:
        if letter.isalpha():
            return True
    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # 1. Go through first level in the tree
    # 2. If any of them are NPs with Ns as their next child, print them
    # 3. Repeat steps one and two with the subtrees until there are no more subtrees to compute
    nChunks = []

    parentedTree = nltk.tree.ParentedTree.convert(tree)

    for word in parentedTree.subtrees():
        if word.label() == "N":
            nChunks.append()

    return nChunks


if __name__ == "__main__":
    main()
