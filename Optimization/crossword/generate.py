import sys
import copy

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox(
                            (0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        # Checks if a word in a variable's domain isn't the variable's length.
        for var in self.crossword.variables:
            for word in copy.deepcopy(self.domains[var]):
                # If the word isn't the same length, it removes the word from the domain.
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        # Gets the overlap between the two variables.
        overlap = self.crossword.overlaps[x, y]

        # If there is no overlap we don't need to do anything so we return no changes made.
        if overlap == None:
            return False

        # Goes through each word in x's domain and checks to see if there is a coresponding word in y's domain that satisfies the overlap condition.
        madeChanges = False
        foundAWord = True
        for x_word in copy.deepcopy(self.domains[x]):
            foundAWord = False
            for y_word in self.domains[y]:
                # If the overlap is the same letter, then it works and we can move onto the next word.
                if x_word[overlap[0]] == y_word[overlap[1]]:
                    foundAWord = True
                    break
            # If there was no coresponding word, we remove that word from x's domain, and we update that we have made changes to x's domain.
            if not foundAWord:
                madeChanges = True
                self.domains[x].remove(x_word)

        return madeChanges

    def ac3(self, arcs=None):
        """Updates `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begins with initial list of all arcs in the problem.
        Otherwise, uses `arcs` as the initial list of arcs to make consistent.

        Returns:
            bool: Returns `True` if arc consistency is enforced and no domains are empty;
        returns `False` if one or more domains end up empty.
        """

        # If the arcs haven't been supplied to us, we use every arc in the CSP.
        if arcs == None:
            # Goes through each variable in the crossword and add it and its neighbour to the queue
            queue = list()
            for var in self.crossword.variables:
                neighbours = self.crossword.neighbors(var)
                for neighbour in neighbours:
                    queue.append((var, neighbour))
        # If there are arcs supplied, we just use them as the queue.
        else:
            queue = arcs

        # Main AC-3 algorithm.
        while queue != list():
            # Takes an arc and checks if it is arc-consistent. If it isn't, it makes it arc consistent then checks all its neighbours again to see if no problems have arised.
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                # If the domain is empty then there is no solutions to the problem and we return False.
                if self.domains[x] == set():
                    return False
                for neighbour in self.crossword.neighbors(x) - {y}:
                    queue.append((neighbour, x))

        # Once the queue is empty we know we have solved the problem and we return True.
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Gets all the variables
        variables = self.crossword.variables

        # Goes through and checks if there is a value associated with each variable.
        for var in variables:
            # If there isn't a value, returns False
            if assignment[var] == None:
                return False

        # If there is a value for each variable returns True
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        variables = list(assignment.keys())

        usedWords = set()

        for var in variables:
            if assignment[var] != None:
                if assignment[var] in usedWords:
                    return False
                if len(assignment[var]) != var.length:
                    return False
                neighbours = self.crossword.neighbors(var)
                for neighbour in neighbours:
                    if assignment[neighbour] != None:
                        overlap = self.crossword.overlaps[var, neighbour]
                        if assignment[var][overlap[0]] != assignment[neighbour][overlap[1]]:
                            return False

                usedWords.add(assignment[var])

        return True

    def order_domain_values(self, var, assignment):
        """Returns a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, will be the one
        that rules out the fewest values among the neighbors of `var`.

        Returns:
            list: (see above)
        """
        # Gets all the neighbours and all the variables that have already been assigned values
        neighbours = self.crossword.neighbors(var)
        alreadyAssigned = self.getAlreadyAssigned(assignment)

        wordsValues = dict()

        # Gets all the words that are able to be used by var
        words = self.domains[var]

        # Goes through each word and checks how many words in each neighbours domain would be unuseable if that word were to be used.
        for word in words:
            wordVal = 0
            for neighbour in neighbours:
                # If the neighbour already has a value assigned to it there is no point in checking because the outcome would be the same regardless.
                if not neighbour in alreadyAssigned:
                    overlap = self.crossword.overlaps[var, neighbour]
                    neighbourWords = self.domains[neighbour]

                    # Counts how many words in the domain don't match
                    for neighbourWord in neighbourWords:
                        if word[overlap[0]] != neighbourWord[overlap[1]]:
                            wordVal += 1

            # That count is then assigned to that word and recorded in the words values dictionary.
            wordsValues[word] = wordVal

        # Sorts all the items based on the values and returns that sorted list.
        return self.sortBasedOnDict(wordsValues)

    def select_unassigned_variable(self, assignment):
        """Returns an unassigned variable not already part of `assignment`.
        Chooses the variable with the minimum number of remaining values
        in its domain. If there is a tie, chooses the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.

        Returns:
            class Variable: (see above)
        """

        # Finds the variable(s) with the least amount of words remaining in the domain.
        smallestDomain = set()
        smallestDomainVal = float('inf')
        assignedVariables = self.getAlreadyAssigned(assignment)
        for var in self.crossword.variables:
            # If the variable has already been assigned a value, we don't need to do anything to it so we skip it.
            if not var in assignedVariables:
                if len(self.domains[var]) < smallestDomainVal:
                    smallestDomainVal = len(self.domains[var])
                    smallestDomain.clear()
                    smallestDomain.add(var)

                elif len(self.domains[var]) == smallestDomainVal:
                    smallestDomain.add(var)

        # If there is only one variable with the smallest number (a.k.a no tie), returns that variable.
        if len(smallestDomain) == 1:
            return smallestDomain.pop()

        # If there is a tie, returns the variable with the most neighbours.
        # Ties don't matter, so we just arbitrarly choose one.
        mostNeighbours = None
        mostNeighboursNum = float('-inf')
        for var in smallestDomain:
            if len(self.crossword.neighbors(var)) > mostNeighboursNum:
                mostNeighboursNum = len(self.crossword.neighbors(var))
                mostNeighbours = var

        return mostNeighbours

    def backtrack(self, assignment):
        """

        """
        """Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.

        Returns:
            None: Returns None if no assignment is possible
            Dict: Returns `assignment` which is a mapping from variables to words.
        """
        # If the assignment is empty, we make every variable map to None
        if assignment == dict():
            for var in self.crossword.variables:
                assignment[var] = None

        # If the assignment is complete we return the assignment
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        # Recursively checks if every neighbor is consistent with each other and if each assignment works.
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result != None:
                    return result
                assignment[var] = None
            else:
                assignment[var] = None
        return None

    def isAssignementComplete(self, assignment):
        keys = assignment.keys()

        for key in keys:
            if assignment[key] == None:
                return False

        return True

    def sortBasedOnDict(self, dictionary: dict):
        """Takes a dictionary and returns the keys sorted from least to greatest
        based on their integer values.

        Args:
            dictionary (dict): Dictionary where the keys are strings and values are integers.

        Returns:
            list: list of the keys sorted based on their values from least to greatest.
        """
        # Sort the dictionary values from least to greatest
        itemsSorted = sorted(dictionary.items(), key=lambda item: item[1])

        # Adds all the keys to the list based on the order that their values were sorted
        keysSorted = []
        for item in itemsSorted:
            keysSorted.append(item[0])

        return keysSorted

    def getAlreadyAssigned(self, assignment):
        keys = assignment.keys()

        alreadyAssigned = set()

        for key in keys:
            if assignment[key] != None:
                alreadyAssigned.add(key)

        return alreadyAssigned


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
