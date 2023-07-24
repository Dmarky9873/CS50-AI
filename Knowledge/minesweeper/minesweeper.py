import itertools
import random
import copy
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """Returns the set of all cells in self.cells known to be mines.

        Returns:
            set: set of cells (tuple of x and y coordinates) that are mines for sure.
        """
        # If the amount of mines is equal to the amount of cells, then we know for sure that all the cells within the set are mines.
        if len(self.cells) == self.count:
            return self.cells

    def known_safes(self):
        """Returns the set of all cells in self.cells known to be safe.

        Returns:
            set: set of cells (tuple of x and y coordinates) that are not mines for sure
        """
        # If there are no mines within the set then we know for sure that all the cells are safe within the set.
        if self.count == 0:
            return self.cells

    def mark_mine(self, cell):
        """Updates internal knowledge representation given the fact that
        a cell is known to be a mine.

        Args:
            cell (tuple): takes a cell and removes it from the set of all unsure cells as it is known to be a mine.
        """
        # If the cell is a mine, then we can remove it from the set of unsure cells and decrease the count of mines within that set.
        self.cells.remove(cell)
        self.count -= 1

    def mark_safe(self, cell):
        """Updates internal knowledge representation given the fact that
        a cell is known to be safe.

        Args:
            cell (tuple): takes a cell and removes it from the set of all unsure cells as it is known to be safe.
        """

        # If the cell isn't a mine, we can remove it from the unsure cells as we know that it is safe. We do not modify count as it isn't a mine.
        self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.

        Args:
            cell (tuple): a cell that is known to be a mine, will be used to then update all knowledge that this is a mine from now on.
        """
        # Adds the cell to the set of all known mines
        self.mines.add(cell)

        # Tells all sentences that contain a given cell that it is a mine.
        for sentence in self.knowledge:
            if cell in sentence.cells:
                sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.

        Args:
            cell (tuple): a cell that is known to be safe, will be used to then update the knowledge that it is safe.
        """
        # Adds the cell to the set of all known safe cells
        self.safes.add(cell)

        # Tells all sentences that contain a given cell that it is safe.
        for sentence in self.knowledge:
            if cell in sentence.cells:
                sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function will:
            1) mark the cell as a move that has been made\n
            2) mark the cell as safe\n
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`\n
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base\n
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge\n

        Args:
            cell (`tuple`): the cell that we are talking about; we know how many neighbouring cells have mines in them.
            count (`int`): the integer amount of neighbouring cells that have mines in them.
        """

        # ! Marks the cell as a move
        self.moves_made.add(cell)

        # ! Marks the cell as safe
        self.mark_safe(cell)

        # ! Adds a new sentence to the AI's KB because we know have new info, we know how many mines there are for a set of 8 cells.
        # Gets all the neighbours for a cell
        neighbours = self.get_neighbours(cell)
        # Creates a deep copy of the count because we will need to submit it when creating a new sentence, plus we are changing it when doing calculations based on info we already have in our KB.
        new_count = copy.deepcopy(count)

        # Iterates through each neighbour and checks to see if it is already known to be safe or a mine, thereby allowing us to remove it from our set of neighbours, helping our info.
        for neighbour in copy.deepcopy(neighbours):
            if neighbour in self.mines:
                neighbours.remove(neighbour)
                new_count -= 1
            elif neighbour in self.safes:
                neighbours.remove(neighbour)

        # Creates a new sentence based on the information we generated and adds it to the KB.
        sentence = Sentence(neighbours, new_count)
        self.knowledge.append(sentence)

        # ! Marks any additional cells as safes or mines if it can be concluded based on the AI's knowledge base now that we have new information.

        # Iterates through each sentence in the KB and checks to see if we have new mines or safes to add to our sets with the info we added.
        for sentence in self.knowledge:
            # Gets all the mines and safes we know now with the new info.
            mines = copy.deepcopy(sentence.known_mines())
            safes = copy.deepcopy(sentence.known_safes())

            # Tells all the sentences this new information of the new mines and safes.
            self.update_mines_safes(mines, safes)

        # ! Makes inferences based on existing knowledge know that we have more information.
        # Checks to see if two sentences are subsets, and if they are, creates a new sentence based on that information
        # Example: if we know set {A, B, C, D} has 3 mines, and we know that set {C, D} has one mine, then we know that set {A, B} has 2 mines, and by extension, both A and B are mines!
        for sentence_A in self.knowledge:
            for sentence_B in self.knowledge:
                # Checks to see if they are subsets
                if sentence_A.cells.issubset(sentence_B.cells):
                    # Creates a new sentence with the information we can attain based on the combination of subsets.
                    new_sentence = Sentence(
                        sentence_B.cells - sentence_A.cells, sentence_B.count - sentence_A.count)

                    # Goes through and checks to see if we have any new mines or safes based on the new information we have gathered.
                    mines = new_sentence.known_mines()
                    safes = new_sentence.known_safes()

                    # Updates all sentences with the new information
                    self.update_mines_safes(mines, safes)

    def update_mines_safes(self, mines, safes):
        """Takes two sets, mines and safes and 
        tells each sentence that they are mines
        and safes.

        Args:
            mines (set): set of mines
            safes (set): set of safes
        """
        # If there are mines pr safes to add, tells all sentences about them.
        if mines:
            for mine in mines:
                self.mark_mine(mine)

        if safes:
            for safe in safes:
                self.mark_safe(safe)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for move in self.safes:
            if not move in self.moves_made:
                return move

        return None

    def get_possible_moves(self):
        """Generates all possible moves that is not guarenteed to be a mine and hasn't been tried before.

        Returns:
            set: Returns a set of all possible moves as stated above. Returns None if no possible moves are available given the above criteria.
        """

        # Creates an empty set
        moves = set()

        # Goes through every cell and checks to see if it has been tried before and if it is a known mine. If it is neither adds it to he moves set.
        for x in range(self.height):
            for y in range(self.width):
                currCell = (x, y)
                if (not currCell in self.moves_made) and (not currCell in self.mines):
                    moves.add(currCell)

        # If we have found at least one move that meets the criteria then we return it if not return None.
        if len(moves) > 0:
            return moves
        else:
            return None

    def make_random_move(self):
        """Returns a move to make on the Minesweeper board.
        Randomly chooses among cells that:
            1) have not already been chosen, and \n
            2) are not known to be mines

        Returns:
            tuple: a cell (x, y coordinate) that has been randomly chosen out of the possible moves on the board.
        """
        # Gets possible moves based on the criteria
        moves = self.get_possible_moves()

        # If there are moves then it returns a random choice from that set, if not it returns None
        if moves:
            return random.choice(list(moves))
        else:
            return None

    def get_neighbours(self, cell):
        """Takes a cell and returns a set of all of its neighbours not including itself that are one cell away

        Args:
            cell (tuple): Tuple of the x and y coordinates of a given cell.

        Returns:
            set: A set of cells (x and y coordinates), with each cell being 1 away from the argument cell.
        """
        neighbours = set()

        # Gets each cell that is one cell away
        for x in range(self.height):
            for y in range(self.width):
                if abs(cell[0] - x) <= 1 and abs(cell[1] - y) <= 1 and (x, y) != cell:
                    neighbours.add((x, y))
        return neighbours
