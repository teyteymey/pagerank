import os
import random
import re
import sys
from collections import defaultdict
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


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # probability of following the link from page
    linked = corpus[page]
    probabilities = {}

    for page in corpus:
        # probability of choosing random between all pages
        probabilities[page] = (1-damping_factor)/len(corpus)
        # meaning it has direct connection
        if page in linked:
            probabilities[page] += len(linked)/damping_factor
            
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank = {}
    page = random.choice(list(corpus.keys()))
    for i in range(n):
        resulting_model = transition_model(corpus, page, damping_factor)
        # choose one of the pages having in mind their weights
        page = random.choices(list(resulting_model.keys()), weights=list(resulting_model.values()), k=1)[0]
        # returns a list so we fetch the element
        rank[page] = rank.get(page, 0) + 1

    for page in rank:
        rank[page] /= n

    return rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialize to equal probability
    rank = {page: 1/len(corpus) for page in corpus}
    # reverse the corpus to know which pages could lead to key
    reverse_corpus = defaultdict(list)
    print(corpus)
    for page in corpus:
        # A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).
        if not corpus[page]:
            print("LINKS TO NOTHING")
            for linked_page in corpus.keys():
                reverse_corpus[linked_page].append(page)
        else:    
            for linked_page in corpus[page]:
                reverse_corpus[linked_page].append(page)
    print(reverse_corpus)
        
    #corpus - page1 links to page3 page4 page 5, page2 links to none
    n_pages = len(corpus)
    old_rank = copy.deepcopy(rank)
    changes = True
    safety = 0
    while changes or safety > 1000:
        # default false but set to true if one of the pages doesnt converge
        changes = False
        # iterate through all pages calculating probability
        for page in rank:
            # get pages that link to current page
            origin_pages = reverse_corpus[page]
            # calculate the prob of landing in it in randon
            prob_random = (1-damping_factor)/n_pages
            # prob of combining origin pages probabilities
            prob_origins = 0
            for link in origin_pages:
                # probability of that origin page 
                prob_origins += old_rank[link] / len(corpus[link])
            rank[page] = prob_random + damping_factor * prob_origins
            if abs(rank[page] - old_rank[page]) > 0.001:
                changes = True
        old_rank = copy.deepcopy(rank)
        
    return rank


if __name__ == "__main__":
    main()
