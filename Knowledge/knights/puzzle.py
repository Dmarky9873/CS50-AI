from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# XOR = And(Or(P, Q)), Not(And(P, Q))


def XOR(P, Q):
    return And(Or(P, Q), Not(And(P, Q)))


# Makes sure that XKnight and XKnave cannot coexist where X is A B and C
default_rules = And(XOR(AKnight, AKnave), XOR(
    BKnight, BKnave), XOR(CKnight, CKnave))


# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # Default rules
    default_rules,

    # This is saying that if A were a knight, A would be both
    # a knight and a knave, but if A were a knave, A would be
    # neither be a knight or a knave
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave)))

    # Logic:
    # AKnight → (AKnight ∧ AKnave)
    # AKnave → ¬(AKnight ∧ AKnave)

    # Hence the only possible conclusion is that A is a Knave
    # because it would be impossible for it to be both!
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # Default rules
    default_rules,

    # This is saying that if A were a Knight then both A and
    # B would have to be Knaves, but if A were a Knave then
    # A would have to be a Knave and B would have to be a
    # Knight
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, And(AKnave, BKnight))

    # Logic:
    # AKnight → (AKnave ∧ BKnave)
    # AKnave → (AKnave ∧ BKnight)

    # Therefore the only reasonable solution would be that A
    # is a Knave and B is a Knight because for A to be Knight
    # Means that it told a lie saying its a Knave, causing a
    # contradiction.
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # Default rules
    default_rules,

    # This is saying that if A were to be a knight then B would
    # have to be a Knight too or A and B have to be Knaves. Or
    # if A is a knave then B would have to be a Knight or A would
    # have to be a Knight and B a Knave. See the obvious
    # contradiction?
    Implication(AKnight, Or(And(AKnave, BKnave), And(AKnight, BKnight))),
    Implication(AKnave, Or(And(AKnave, BKnight), And(AKnight, BKnave))),
    Implication(BKnight, Or(And(AKnave, BKnight), And(AKnight, BKnave))),
    Implication(BKnave, Or(And(AKnave, BKnave), And(AKnight, BKnight)))

    # Logic:
    # AKnight → (AKnave ∧ BKnave) ∨ (AKnight ∧ BKnight)
    # AKnave → (AKnave ∧ BKnight) ∨ (AKnight ∧ BKnave)
    # BKnight → (AKnave ∧ BKnight) ∨ (AKnight ∧ BKnave)
    # BKnave → (AKnave ∧ BKnave) ∨ (AKnight ∧ BKnight)

    # Therefore B has to be the Knight because if A were the Knight then B,
    # saying the obvious lie would contradict the initial statement that
    # A is a Knight
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Default rules
    default_rules,

    # If A is a knight, then B would have to be a Knave because it is impossible
    # for a character to assert that it is a Knave
    Implication(AKnight, BKnave),
    Implication(AKnave, BKnave),

    # If B is a knight then C is a Knave and vice versa
    Implication(BKnight, CKnave),
    Implication(BKnave, CKnight),

    # If C is a Knight then A is a Knight and vice versa
    Implication(CKnight, AKnight),
    Implication(CKnave, AKnave)

    # Logic:
    # AKnight → BKnave
    # AKnave → BKnight
    # BKnight → CKnave
    # BKnave → CKnight
    # CKnight → AKnight
    # CKnave → AKnave

    # Therefore B has to be a Knave because it would cause a contradiction otherwise
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
