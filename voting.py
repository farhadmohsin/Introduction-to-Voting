import numpy as np
from time import time
#%% functions for generating preference profiles
def permutation(lst):
    """
    function to create permutations of a given list
        supporting function for ranking_count
    reference: https://www.geeksforgeeks.org/generate-all-the-permutation-of-a-list-in-python/
    """
    if(len(lst) == 0):
        return []
    if(len(lst) == 1):
        return [lst]
    l = []   
    for i in range(len(lst)): 
       m = lst[i] 
       remLst = lst[:i] + lst[i+1:] 
       for p in permutation(remLst): 
           l.append([m] + p) 
    return l

def gen_random_vote(m):
    # m is the number of alternatives
    # this functions generates an uniformly random profile
    # i.e 1/m! probability of each profile
    
    alts = list(range(m))
    perms = permutation(alts)
    
    t = np.random.randint(0,len(perms))
    
    return perms[t]

def gen_pref_profile(N,m):
    votes = []
    for t in range(N):
        votes.append(gen_random_vote(m))
    return np.array(votes)

#%% functions for calculating winners (multiwinner version)
# TODO: Need to write a multi-round version (have function as parameters)
    
def Copeland_winner(votes):
    """
    Description:
        Calculate Copeland winner given a preference profile
    Parameters:
        votes:  preference profile with n voters and m alternatives
    Output:
        winner: Copeland winner
        scores: pairwise-wins for each alternative
    """
    n,m = votes.shape
    scores = np.zeros(m)
    for m1 in range(m):
        for m2 in range(m1+1,m):
            m1prefm2 = 0        #m1prefm2 would hold #voters with m1 \pref m2
            for v in votes:
                if(v.tolist().index(m1) < v.tolist().index(m2)):
                    m1prefm2 += 1
            m2prefm1 = n - m1prefm2
            if(m1prefm2 == m2prefm1):
                scores[m1] += 0.5
                scores[m2] += 0.5
            elif(m1prefm2 > m2prefm1):
                scores[m1] += 1
            else:
                scores[m2] += 1
    winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
    
    return winner, scores
    
def Borda_winner(votes):
    n, m = votes.shape
    scores = np.zeros(m)
    for i in range(n):
        for j in range(m):
            scores[votes[i][j]] += m-j-1 
    winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
    
    return winner, scores

def plurality_winner(votes):
    n, m = votes.shape
    scores = np.zeros(m)
    for i in range(n):
        scores[votes[i][0]] += 1
    winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
    
    return winner, scores

def STV_helper(votes, n, m, removed):
    """
    Parameters
    ----------
    votes : preference profile
    n : #votes
    m : #candidates in original election
    removed : already removed candidates
    """
    winner, scores = plurality_winner(votes)
    print(winner, scores, removed)
    
    if(np.max(scores) >= n/2):
        return winner, scores
    rest_scores = scores
    rest_scores[removed] = np.inf
    c_last = np.argmin(rest_scores)
    
    removed.append(c_last)
    new_votes = []
    for v in votes:
        newv = np.delete(v, np.where(v==c_last))
        newv = np.append(newv, c_last)
        
        new_votes.append(newv)
    
    return STV_helper(np.array(new_votes), n, m, removed)
    
def STV_winner(votes):
    votes_cpy = votes.copy()
    n, m = votes_cpy.shape
    return STV_helper(votes_cpy, n, m, [])

#%% tiebreakers

def singleton_winner(vote, winners):
    ranks = [np.argwhere(vote == w).flatten()[0] for w in winners]
    return winners[np.argmin(ranks)]

def lexicographic_tiebreaking(votes, winners):
    return np.min(winners)

def voter1_tiebreaking(votes, winners):
    return singleton_winner(votes[0],winners)

#%%
N = 20
m = 4
votes = gen_pref_profile(N, m)

winner, scores = plurality_winner(votes)
winner, scores = Borda_winner(votes)
winner, scores = Copeland_winner(votes)
winner, scores = STV_winner(votes)

