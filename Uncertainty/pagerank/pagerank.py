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
    pages = corpus.keys()
    probabilities = dict()

    # If the page has no links, return all the pages including the page were looking at
    # currently at an equal probability that adds to one.
    if corpus[page] == {}:
        for p in pages:
            probabilities[p] = 1/len(pages)
    else:
        # Sets the current page's probability to be 1 - dampining_factor over the amount of different pages
        probabilities[page] = (1 - damping_factor)/len(pages)
        # Sets the rest of the pages to be an equal amount with dampining factor added
        for p in corpus[page]:
            probabilities[p] = (1/len(list(corpus[page]))) * \
                0.85 + (1 - damping_factor)/len(pages)

    return probabilities


def sample_pagerank(corpus: dict, damping_factor: float, n: int):
    """Returns a PageRank value for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Returns:
        `dict()`: Returns a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Puts all the different pages into a dictionary and sets all the values to 0
    pages = corpus.keys()
    pageRanks = dict()
    for page in pages:
        pageRanks[page] = 0

    # Chooses a random page to start at
    currPage = random.choice(list(pages))

    # Samples through each page N times with damping_factor chance of picking a random page
    for i in range(n):
        # Adds one to the current page that was chosen and preps the next iteration
        pageRanks[currPage] += 1
        model = transition_model(corpus, currPage, damping_factor)
        num = random.random()
        modelPages = model.keys()
        accum = 0

        # Goes through each page within the model and checks whether it was chosen by the random number, if so sets it as the currPage
        for modelPage in modelPages:
            accum += model[modelPage]
            if num <= accum:
                currPage = modelPage
                break

    # Formats the dictionary properly
    for page in pageRanks:
        while str(pageRanks[page])[0] != '0':
            pageRanks[page] = pageRanks[page]/10

    return pageRanks


def iterate_pagerank(corpus: dict, damping_factor: float):
    """
    Returns the PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Returns:
        `dict()`: Returns a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values will sum to 1.
    """
    pages = list(corpus.keys())
    pageRank = dict()
    d = damping_factor
    N = len(pages)

    # Goes through each page and sets their pagerank to be 1/N at the beginning.
    for page in pages:
        pageRank[page] = 1/N

    # Keeps repeating the formula until they converge, increasing accuracy
    while True:
        # Sets a veriable to the previous pagerank to test for convergence at the end
        prevPageRank = copy.deepcopy(pageRank)

        # Goes through each page and does the formula and records the new variable in the pagerank dict
        for page in pages:
            pageRank[page] = ((1 - d)/N) + \
                (d*(pageRankSummation(page, corpus, pageRank)))

        # Checks to see if there is a marginal difference
        marginalDiff = 0.001
        hasMarginalDiff = True
        for page in pages:
            if abs(prevPageRank[page] - pageRank[page]) >= marginalDiff:
                hasMarginalDiff = False

        # If there is a marginal difference, returns the pagerank dictionary
        if hasMarginalDiff:
            return pageRank


def numLinks(page: str, corpus: dict):
    """Gets the amount of links in a given page.

    Args:
        page (`str()`): The page whos number of links you are trying to find.
        corpus (`dict()`): The dictionary of all pages and their links.

    Returns:
        `int()`: The integer number of links on a given page.
    """
    return len(corpus[page])


def pageRankSummation(page: str, corpus: dict, pagerank: dict):
    """Does the summation part of the PageRank formula.

    Args:
        page (`str()`): Page whos Pagerank you are trying to find.
        corpus (`dict()`): Dictionary of all the pages and their links.
        pagerank (`dict()`): Dictionary of PageRank of all the different pages.

    Returns:
        `float()`: Returns the float of the result of the summation.
    """
    accum = 0
    pages = list(corpus.keys())
    pagesThatLink = set()

    # Finds all the pages that link to `page`
    for p in pages:
        if page in corpus[p]:
            pagesThatLink.add(p)

    # Accumulates all the pageranks of the pages and divides them by the number of links within that page.
    for p in pagesThatLink:
        accum += pagerank[p]/numLinks(p, corpus)

    return float(accum)


if __name__ == "__main__":
    main()
