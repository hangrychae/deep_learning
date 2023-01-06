import numpy as np


def clean_path(path):
    """ utility function that performs basic text cleaning on path """

    # No need to modify
    path = str(path).replace("'","")
    path = path.replace(",","")
    path = path.replace(" ","")
    path = path.replace("[","")
    path = path.replace("]","")

    return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        sym_prob, seq_len, batch_size = y_probs.shape

        symbol_sequence = []
        decoded_path = []
        #decoded_path = np.empty(shape=(batch_size,), dtype="str")
        blank = 0
        path_probs = np.ones(shape=(batch_size,))
        prob_sym = "BLANK"
        # max_probs = np.zeros(shape=(len(y_probs[1]),))
        max_prob = 0

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path

        
        for b in range(batch_size): # batch size
            symbol_seq = ""
            for l in range(seq_len): # sequence length 
                max = max_prob
                for p in range(sym_prob): # symbol prob
                    prob = y_probs[p][l][b]
                    if max < prob: # found new max prob
                        if p != 0: 
                            prob_sym = self.symbol_set[p-1] # as no blank 
                        max = prob
                path_probs[b] *= max
                symbol_seq += prob_sym
            prob_sym = "Blank"
            symbol_sequence.append(symbol_seq)
       

        for b in range(batch_size): # batch size
            path = ""
            for l in range(seq_len): # sequence length 
                if l == 0:
                    if symbol_sequence[b][l] != "Blank":
                        path += symbol_sequence[b][l]
                else:
                    if symbol_sequence[b][l] != "Blank" and symbol_sequence[b][l] != symbol_sequence[b][l-1]:
                        path += symbol_sequence[b][l]
            decoded_path.append(path)


        for i in range(len(decoded_path)):
            decoded_path[i] = clean_path(decoded_path[i])
        
        if batch_size == 1:
            return decoded_path[0], path_probs[0]
        else:
            return decoded_path, path_probs
        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
        self.Pathscore, self.BlankPathScore = dict(), dict()

    def InitializePaths(self, symbolset, y):
        InitialBlankPathScore, InitialPathScore = dict(), dict()
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ""
        InitialBlankPathScore[path] = y[0] # Score of blank at t=1
        InitialPathsWithFinalBlank = {path}

        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = set()
        for i, symbset in enumerate(symbolset): # This is the entire symbol set, without the blank
            path = symbset
            InitialPathScore[path] = y[i+1] # Score of symbol c at t=1
            InitialPathsWithFinalSymbol.add(path) # Set addition

        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol,\
                InitialBlankPathScore, InitialPathScore

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore = dict()
        PrunedPathScore = dict()
        scorelist = [None for i in range(len(PathsWithTerminalBlank)+len(PathsWithTerminalSymbol))]
        # First gather all the relevant scores

        i = 0
        for p in PathsWithTerminalBlank:
            scorelist[i] = BlankPathScore[p]
            i += 1
        for p in PathsWithTerminalSymbol:
            scorelist[i] =  PathScore[p]
            i += 1


        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist = sorted(scorelist, reverse=True) # In decreasing order
        if BeamWidth < len(scorelist):
            cutoff = scorelist[BeamWidth]
        else:
            cutoff = scorelist[-1]

        PrunedPathsWithTerminalBlank = set()
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] > cutoff: # lecture slide wrong. should be > 
                PrunedPathsWithTerminalBlank.add(p) # Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]


        PrunedPathsWithTerminalSymbol = set()
        for p in PathsWithTerminalSymbol:
            if PathScore[p] > cutoff:
                PrunedPathsWithTerminalSymbol.add(p) # Set addition
                PrunedPathScore[p] = PathScore[p]

        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = dict()
        # First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)
        for path in PathsWithTerminalBlank:
            # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank.add(path) # Set addition
            UpdatedBlankPathScore[path] = self.BlankPathScore[path] * y[0]
        
        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank# simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += self.Pathscore[path] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.add(path) # Set addition
                UpdatedBlankPathScore[path] = self.Pathscore[path] * y[0]
    
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore
    
    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
        UpdatedPathsWithTerminalSymbol = set()
        # UpdatedPathScore = []
        UpdatedPathScore = dict()
        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
                newpath = path + c # Concatenation
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = self.BlankPathScore[path] * y[i+1]

        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
                for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
                    if c == path[-1]: # Horizontal transitions don’t extend the sequence
                        newpath = path
                    else:
                        newpath = path + c
    
                    if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                        UpdatedPathScore[newpath] += self.Pathscore[path] * y[i+1]
                    else: # Create new path
                        UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                        UpdatedPathScore[newpath] = self.Pathscore[path] * y[i+1]

        return  UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def MergeIdenticalPaths(self, NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore):
        # All paths with terminal symbols will remain
        MergedPaths = NewPathsWithTerminalSymbol
        FinalPathScore = NewPathScore

        # Paths with terminal blanks will contribute scores to existing identical paths from 
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in NewPathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += NewBlankPathScore[p]
            else:
                MergedPaths.add(p) # Set addition
                FinalPathScore[p] = NewBlankPathScore[p]

        return MergedPaths, FinalPathScore

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        sym_prob, seq_len, batch_size = y_probs.shape
        
        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None


        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return      

    
        BestPath, FinalPathScore = [], []

        for b in range(batch_size):
            NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore \
            = self.InitializePaths(self.symbol_set, y_probs[:, 0, b])
            best_path, merged_path_scores = None, None
            for l in range(seq_len-1):
                # Prune the collection down to the BeamWidth
                PathsWithTerminalBlank, PathsWithTerminalSymbol, self.BlankPathScore, self.Pathscore \
                = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, self.beam_width)

                # First extend paths by a blank
                NewPathsWithTerminalBlank, NewBlankPathScore \
                = self.ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, l+1, b])

                # Next extend paths by a symbol
                NewPathsWithTerminalSymbol, NewPathScore \
                = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.symbol_set, y_probs[:, l+1, b])


            # Merge identical paths differing only by the final blank
            MergedPaths, merged_path_scores \
            = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        
            # Pick best path
            best_path = max(merged_path_scores, key=lambda x: merged_path_scores[x]) # Find the path with the best score

            BestPath.append(best_path)
            FinalPathScore.append(merged_path_scores)

        if batch_size == 1:
            return BestPath[0], FinalPathScore[0]
        else:
            return BestPath, FinalPathScore
        
