"""_summary_

Raises:
    NotImplementedError: _description_
    NotImplementedError: _description_
    NotImplementedError: _description_

Returns:
    _type_: _description_
"""


import csv
import itertools
import sys

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
    """_summary_
    """

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
    joint_prob = 1
    for person in people:
        person_prob = 1
        person_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        person_trait = person in have_trait
        person_mother = people[person]['mother']
        person_father = people[person]['father']
        if not person_father and not person_mother:
            person_prob *= PROBS["gene"][person_genes]
        else:
            prob_mother_giving = find_prob_inherit(
                person_mother, one_gene, two_genes)
            prob_father_giving = find_prob_inherit(
                person_father, one_gene, two_genes)

            if person_genes == 1:
                person_prob *= prob_father_giving * \
                    (1-prob_mother_giving) + \
                    prob_mother_giving*(1 - prob_father_giving)
            elif person_genes == 2:
                person_prob *= prob_father_giving*prob_mother_giving
            else:
                person_prob *= (1 - prob_father_giving) * \
                    (1 - prob_mother_giving)

        person_prob *= PROBS['trait'][person_genes][person_trait]

        joint_prob *= person_prob

    return joint_prob


def find_prob_inherit(parent_name, one_gene, two_genes):
    """_summary_

    Args:
        parent_name (_type_): _description_
        one_genes (_type_): _description_
        two_genes (_type_): _description_

    Returns:
        _type_: _description_
    """
    if parent_name in one_gene:
        return 0.5
    elif parent_name in two_genes:
        return 1 - PROBS['mutation']
    return PROBS['mutation']


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
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
    Update `probabilities` such that each probability distribution
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


if __name__ == "__main__":
    main()
