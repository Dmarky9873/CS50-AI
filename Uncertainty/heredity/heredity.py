import csv
import itertools
import sys
import random

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    no_genes = []
    names = list(people.keys())
    for name in names:
        if (name not in one_gene) and (name not in two_genes):
            no_genes.append(name)

    odds = set()

    for person in one_gene:
        if person in have_trait:
            hastrait = True
        else:
            hastrait = False
        if hasParents(people, person):
            parents = getParents(people, person)
            parentsOdds = getParentsOdds(parents, one_gene, two_genes)
            oddsOfGettingOne = parentsOdds['mother-giving']*parentsOdds['father-not-giving'] + \
                parentsOdds['father-giving']*parentsOdds["mother-not-giving"]

            odds.add(oddsOfGettingOne*PROBS["trait"][1][hastrait])
        else:
            odds.add(PROBS["gene"][1]*PROBS["trait"][1][hastrait])

    for person in two_genes:
        if person in have_trait:
            hastrait = True
        else:
            hastrait = False
        if hasParents(people, person):
            parents = getParents(people, person)
            parentsOdds = getParentsOdds(parents, one_gene, two_genes)
            oddsOfGettingTwo = parentsOdds['mother-giving'] * \
                parentsOdds['father-giving']

            odds.add(oddsOfGettingTwo*PROBS['trait'][2][hastrait])
        else:
            odds.add(PROBS["gene"][2]*PROBS["trait"][2][hastrait])

    for person in no_genes:
        if person in have_trait:
            hastrait = True
        else:
            hastrait = False
        if hasParents(people, person):
            parents = getParents(people, person)
            parentsOdds = getParentsOdds(parents, one_gene, two_genes)
            oddsOfGettingNone = parentsOdds['mother-not-giving'] * \
                parentsOdds['father-not-giving']

            odds.add(oddsOfGettingNone*PROBS['trait'][0][hastrait])
        else:
            odds.add(PROBS["gene"][0]*PROBS["trait"][0][hastrait])

    accum = 1
    for num in odds:
        accum *= num

    return accum


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Adds to `probabilities` a new joint probability `p`.
    Each person will have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Gets the list of names of all people.
    people = list(probabilities.keys())

    # Goes through adding p to the required places.
    for person in people:
        if person in one_gene:
            if probabilities[person]["gene"][0] != None:
                probabilities[person]["gene"][1] += p
            else:
                probabilities[person]["gene"][0] = p
        elif person in two_genes:
            if probabilities[person]["gene"][0] != None:
                probabilities[person]["gene"][2] += p
            else:
                probabilities[person]["gene"][0] = p
        else:
            if probabilities[person]["gene"][0] != None:
                probabilities[person]["gene"][0] += p
            else:
                probabilities[person]["gene"][0] = p

        if person in have_trait:
            if probabilities[person]["trait"][True] != None:
                probabilities[person]["trait"][True] += p
            else:
                probabilities[person]["trait"][True] = p
        else:
            if probabilities[person]["trait"][False] != None:
                probabilities[person]["trait"][False] += p
            else:
                probabilities[person]["trait"][False] = p


def normalize(probabilities):
    """
    Updates `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Gets the list of all people in the dict.
    people = list(probabilities.keys())

    # Goes through each person and makes all the probabilities sum to one while keeping the proportions the same.
    for person in people:
        # How we do this is by dividing the value of each probability by the sum of all probabilies in that category.
        geneSum = probabilities[person]["gene"][0] + \
            probabilities[person]["gene"][1] + probabilities[person]["gene"][2]

        # We also round each value to the nearest 4th decimal place to take care of floating point errors.
        probabilities[person]["gene"][0] = round(
            probabilities[person]["gene"][0]/geneSum, 4)
        probabilities[person]["gene"][1] = round(
            probabilities[person]["gene"][1]/geneSum, 4)
        probabilities[person]["gene"][2] = round(
            probabilities[person]["gene"][2]/geneSum, 4)

        traitSum = probabilities[person]["trait"][True] + \
            probabilities[person]["trait"][False]

        probabilities[person]["trait"][True] = round(
            probabilities[person]["trait"][True]/traitSum, 4)
        probabilities[person]["trait"][False] = round(
            probabilities[person]["trait"][False]/traitSum, 4)


def hasParents(people: dict, person: str):
    """Tells you whether or not `person` has parents.

    Args:
        `people` (dict): Dictionary of all people.
        `person` (str): Person we are looking at

    Returns:
        bool: True if `person` has parents and False if otherwise
    """
    if people[person]['mother'] == None and people[person]['father'] == None:
        return False
    return True


def getParents(people: dict, person: str):
    """Gets `person`'s parents.

    Args:
        people (dict): Dictionary of all people.
        person (str): Person we are looking at

    Returns:
        dict: Returns a dictionary with `'mother'` being mapped to `person`'s mother and `'father'` being mapped to `person`'s father.
    """
    return {'mother': people[person]['mother'], 'father': people[person]['father']}


def getParentsOdds(parents: dict, one_gene: list, two_genes: list):
    """Gets the odds of a parent giving and not giving a gene.

    Args:
        parents (dict): Dictionary of the names of a person's mother and father.
        one_gene (list): The list of all people who have one gene.
        two_genes (list): The list of all people who have two genes.

    Returns:
        dict: Returns a dictionary mapping the parent's odds to them.
    """
    if parents['mother'] in two_genes:
        oddsOfMomGiving = 1 - PROBS['mutation']
        oddsOfMomNotGiving = 1 - oddsOfMomGiving
    elif parents['mother'] in one_gene:
        oddsOfMomGiving = 0.5
        oddsOfMomNotGiving = 1 - oddsOfMomGiving
    else:
        oddsOfMomGiving = PROBS['mutation']
        oddsOfMomNotGiving = 1 - oddsOfMomGiving

    if parents['father'] in two_genes:
        oddsOfDadGiving = 1 - PROBS['mutation']
        oddsOfDadNotGiving = 1 - oddsOfDadGiving
    elif parents['father'] in one_gene:
        oddsOfDadGiving = 0.5
        oddsOfDadNotGiving = 1 - oddsOfDadGiving
    else:
        oddsOfDadGiving = PROBS['mutation']
        oddsOfDadNotGiving = 1 - oddsOfDadGiving

    return {'father-giving': oddsOfDadGiving, 'father-not-giving': oddsOfDadNotGiving, 'mother-giving': oddsOfMomGiving, 'mother-not-giving': oddsOfMomNotGiving}


if __name__ == "__main__":
    main()
