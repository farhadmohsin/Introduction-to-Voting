import numpy as np
from itertools import permutations
from collections import defaultdict
from pprint import pprint

def sample_urns_profile(n, m, alpha = 0):
    """
    Parameters
    ----------
    n: no. samples
    m : no. alternatives
    alpha : float, decides how many copies to put bac after drawing each sample.
    alpha = 0 -> IC, alpha = 1/m! -> IAC, high alpha means more correlation

    Returns
    -------
    a preference profile with n rankings

    """
    A = np.arange(m)
    perms = [list(P) for P in list(permutations(A))]
    
    fact_m = len(perms)
    prob = np.ones(fact_m) / fact_m
    
    votes = []
    for i in range(n):
        ranking = np.random.choice(fact_m, p = prob)
        votes.append(perms[ranking])
        prob[ranking] += alpha * fact_m
        prob = prob / np.sum(prob)
    
    return np.array(votes)

def single_peaked(m):
    """
    Parameters
    ----------
    m : no. of alternatives

    Returns
    -------
    return a single single peaked preference
    """
    
    A = np.arange(m)
    
    vote = []
    # sample top alternative
    c = np.random.choice(m)
    vote.append(c)
    min_c = c
    max_c = c
    
    no_sampled = 1
    while no_sampled < m:
        
        if min_c == 0:
            vote.append(max_c + 1)
            max_c += 1
        elif max_c == m - 1:
            vote.append(min_c - 1)
            min_c -= 1
        else:
            c = np.random.choice([min_c - 1, max_c + 1])
            vote.append(c)
            if c == min_c - 1:
                min_c -= 1
            else:
                max_c += 1
            
        no_sampled += 1
    
    return vote

def single_peaked_profile(n, m):
    return np.array([single_peaked(m) for i in range(n)])

def get_histogram(votes):
    cnt = defaultdict(int)
    for R in votes:
        cnt[tuple(R)] += 1
    return cnt

def Gaussian_voters_candidates(n, m, d=2):
    """
    Description:
        Generate m candidates and random voters
        voters are from the same multivar Gaussian
    Parameters:
        n:  voters
        m:   number of alternatives
        d:   dimension of the metric space
    """
    candidates = np.random.uniform(size = [m, d]) #get m candidates
    
    center = np.random.uniform(size = d) # get center for voters
    # covariance matrix is just diagonal with 0.2 variance in all dims
    g = np.random.multivariate_normal(center, np.eye(d)*0.02, size = n)
    
    return g, candidates

def uniform_voters_candidates(n, m, d=2):
    """
    Description:
        Generate m candidates and random voters
        voters are uniformly drawn
    Parameters:
        n:  voters
        m:   number of alternatives
        d:   dimension of the metric space
    """
    candidates = np.random.uniform(size = [m, d]) #get m candidates
    g = np.random.uniform(size = [n, d])
    
    return g, candidates


def random_pref_profile(g, candidates):    
    votes = []
    for agent in g:
        dist = np.linalg.norm(agent - candidates, axis = 1)
        votes.append(np.argsort(dist))
    
    return np.array(votes)
    
def euclidean_profile(n, m, model = 'Gaussian'):
    if model == 'Gaussian' or model == 'G':
        g, candidates = Gaussian_voters_candidates(n, m)
    else:
        g, candidates = uniform_voters_candidates(n, m)
    votes = random_pref_profile(g, candidates)
    return votes


if __name__ == '__main__':
    N = 1000
    m = 3
    
    # votes = sample_urns_profile(N, m, alpha = 0.01)
    # votes = single_peaked_profile(N, m)
    votes = euclidean_profile(N, m, 'U')
    pprint(get_histogram(votes))
