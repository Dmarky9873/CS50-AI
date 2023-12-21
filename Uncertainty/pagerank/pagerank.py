import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: dict, page: str, damping_factor: float):
    """Will return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor` will choose a link at random
    linked to by `page`. With probability `1 - damping_factor` will choose
    a link at random chosen from all pages in the corpus.

    Returns:
        `dict()`: The return value of the function will be a Python 
        dictionary with one key for each page in the corpus. Each key 
        will be mapped to a value representing the probability that
        a random surfer would choose that page next. The values in this 
        returned probability distribution will sum to 1.
    """
    # Gets the name of all the pages in the corpus and creates an empty dictionary
    probabilities = dict()

    # If the page has no links, return all the pages including the page were looking at
    # currently at an equal probability that adds to one.
    if corpus[page] is None:
        for p in corpus:
            probabilities[p] = 1/len(corpus)
    else:
        # Sets the current page's probability to be 1 - dampining_factor over the amount of different pages
        for p in corpus:
            probabilities[p] = (1 - damping_factor)/len(corpus)
        # Sets the rest of the pages to be an equal amount with dampining factor added
        for p in corpus[page]:
            probabilities[p] += damping_factor/len(corpus[page])

    return probabilities


def sample_pagerank(corpus: dict, damping_factor: float, n: int):
    """
    Returns a PageRank value for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Returns:
        `dict()`: Returns a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    probabilities = dict()

    for page in corpus.keys():
        probabilities[page] = 0

    curr_page = random.choice(list(corpus.keys()))

    for i in range(n):
        model = transition_model(corpus, curr_page, damping_factor)

        for page in probabilities:
            probabilities[page] = (i*probabilities[page] + model[page])/(i + 1)

        curr_page = random.choices(
            list(probabilities.keys()), weights=list(model.values()), k=1)[0]

    return probabilities


def iterate_pagerank(corpus: dict, damping_factor: float):
    """
    Returns the PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Returns:
        `dict()`: Returns a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values will sum to 1.
    """
    is_marg_diff = False
    marg_diff = 0.001
    pages = list(corpus.keys())
    probabilities = dict()
    for page in pages:
        probabilities[page] = 1/len(pages)

    while not is_marg_diff:
        old_probabilities = copy.deepcopy(probabilities)

        for page in probabilities:
            probabilities[page] = PR(page, damping_factor, len(
                pages), probabilities, corpus)

        for page, probability in probabilities.items():
            is_marg_diff = True
            if abs(probability - old_probabilities[page]) > marg_diff:
                is_marg_diff = False

    return probabilities


def PR(page, d, n, probabilities, corpus):
    return (1 - d)/n + d*(summation(page, corpus, probabilities))


def summation(page, corpus, probabilities):
    result = 0
    for page in pages_that_link(page, corpus):
        result += probabilities[page]/len(corpus[page])
    return result


def pages_that_link(page, corpus):
    """Returns all the pages that link to `page` in `corpus`.

    Args:
        `page` (`str`): The name of the page whos pages who link we are trying to find.
        `corpus` (`dict`): Dictionary of pages and their relationships.

    Returns:
        `set`: The set of all pages that link to `page`.
    """
    pages = set()
    for p in corpus.keys():
        if page in corpus[p]:
            pages.add(p)
    return pages


if __name__ == "__main__":
    main()
