"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

from typing import Union, List, Optional, Callable
from scipy.stats import rankdata
import numpy as np 

def numberfy(x):
    ''' if x could be a number return it as numeric even if it is in string format '''
    try:
        nx=float(x)
    except:
        nx=np.NaN
    return(nx)

def adjusted_FDR_pvalues(unadjusted_p_values: Union[np.ndarray, List[float]]) -> np.ndarray:
                        
    """ adjusts p-values for FDR using the standard Benjamini-Hochberg (Stepup) procedure
    
    Benjamini Y., Hochberg Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing, Journal of the Royal Statistical Society B, 289-300.

    Note: one can use 
     import statsmodels.stats.multitest as smt
     adjusted_pvalues = smt.multipletests(pvals=(0.05, 0.01, 0.1), method='fdr_bh', is_sorted=False, returnsorted=False)[1]
     however, this does not allow for missing values which could occure is using semopy to estimate an SEM with a latent node

    :param unadjusted_p_values: An array or a list of p-values. [could include NANs]
    
    :return: Adjusted FDR p-Values in same order as the input pvaules. 

    """

    mp=len(unadjusted_p_values)
    p_valsArr = np.array(unadjusted_p_values)

    #before starting ensure numeric values and just count them - i.e. exlude null values
    m=0
    p_vals = np.empty(mp)
    for i in range(mp):
        p_vals[i]=numberfy(p_valsArr[i])
        if p_vals[i] >= 0:
            m=m+1
        
    #calculate FDR adjusted pvalues
    ranked_p_values = rankdata(p_vals)
    adjfdr = p_vals * m / ranked_p_values
    adjfdr[adjfdr > 1] = 1
    
    #remember if (k) is rejected (1)..(k) are rejected - the FDR adjustemnt shoud reflect that
    IndexRanks = p_vals.argsort()
    minAP=1
    for i in range(mp-1,-1,-1):
        pointer = int(IndexRanks[i])
        if adjfdr[pointer] > minAP:
            adjfdr [pointer] = minAP
        else:
            minAP = adjfdr[pointer]
    return adjfdr


def adjusted_FDCR_pvalues(unadjusted_p_values,
                          belief_scores,
                          intersection_belief_score=10000.0
                          )-> np.ndarray:
    """ Adjusts p-values for FDR using the Benjamini-Kling Weighted FDCR control adding the Simes family statistic
    
    Benjamini Y., Hochberg Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing, Journal of the Royal Statistical Society B, 289-300.
    Benjamini, Y., Hochberg, Y. (1997). "Multiple hypotheses testing with weights", Scandinavian Journal of Statistics, vol 24, 3, pp 407-418.
    Benjamini Y, Kling Y. E. (2005) A Cost-Base approach to Multiplicity Control, technical paper 
        [can be downloaded at https://docs.google.com/a/businessken.com/viewer?a=v&pid=sites&srcid=YnVzaW5lc3NrZW4uY29tfGhvbWV8Z3g6MTMwODI4YmVmNjI5ZTc4Yg ]

    the convertion of the belief scores to costs as per the original paper above is chosed to be the reciprocal.

    :param unadjusted_p_values: An array or a list of p-values. [could include NANs]
    :param belief_scores: - vector of, non-negaive, belief Scores corresponding to the p-values
                             Inf=the SME thinks the null hypothesiss is absolutly wrong = there is definitely something in it,
                             0 = the SME belives the null hypothesis is an absolute truth - there is nothing to this claim
                             thus the costs are the inverse of the belief
                   these are relative beliefs. if all are equal then we are back to the regular BH procedure [with the adition of the Simes statistic]
                   assume len(unadjusted_p_values) = len(belief_scores) and the elemets are corresponding
    :param intersection_belief_score - belief score for the overall family statistic - Siems
    :return: a vector of length (len (unadjusted_p_values) +1) where
             the first (len (unadjusted_p_values) are  Adjusted FDR p-Values in same order as the input pvaules. 
             the last element is the Simes pvaule
    """

    #checks to implement
    # all pvalues are non negative
    # all belief scores are numbers and are not missing - for now negative and missing beliefs score are assigned a score of 0
    if len(unadjusted_p_values) > len(belief_scores):
        print("there should be a belief score for each p-value")
        return((-17))

    mp=len(unadjusted_p_values)
    p_valsArr = np.array(unadjusted_p_values)
    beliefArr = np.array(belief_scores)

    #before starting ensure numeric values and just count them - i.e. exlude null values
    m=0
    p_vals = np.empty(mp+1) # last element prepared for the Weighted Simes Statistic
    beliefWeights = np.empty(mp+1)
    sumbeliefWeights = 0
    for i in range(mp):
        p_vals[i]        = numberfy(p_valsArr[i])
        beliefWeights[i] = numberfy(beliefArr[i])
        if  beliefWeights[i] >= 0:
            beliefWeights[i] =  1/(beliefWeights[i]+0.00001) # the cost is the reciprocal of the belief + a small constant for zeros

        if p_vals[i] >= 0:
            sumbeliefWeights = sumbeliefWeights + beliefWeights[i]
            m=m+1
        else:
            beliefWeights[i] = 0 # otherwise the cumulative weights will be messed up in following loops
    
    # Weighted Simes Statistic (Benjamini & Hochberg 1997)
    IndexRanks = p_vals.argsort()
    PWeightedSimes = 999999999
    cumulativebeliefWeight = 0
    for i in range(0,mp+1,1):
        pointer = int(IndexRanks[i])
        #dont incluide the place holder for the Simes statistic - we are calcuating it now
        if pointer < mp: 
            cumulativebeliefWeight = cumulativebeliefWeight + beliefWeights[pointer]
            wp = cumulativebeliefWeight / sumbeliefWeights * p_vals[pointer]
            if PWeightedSimes > wp:
               PWeightedSimes = wp  

    #add the Weighted Simes Statistic to the pvalues
    p_vals[mp] = PWeightedSimes
    beliefWeights[mp] = 1/(intersection_belief_score+0.00001) 
    sumbeliefWeights = sumbeliefWeights +  beliefWeights[mp]

    #run through the p-values from biggest to samllest
    IndexRanks = p_vals.argsort()
    cumulativebeliefWeight = sumbeliefWeights
    adjustedFDCR = np.empty(mp+1) #remember, last element is the Weighted Simes
    minAP=1
    for i in range(mp,-1,-1):
        pointer = int(IndexRanks[i])
        #step 1 - calculate the raw FDCR adjestment
        adjustedFDCR[pointer] = sumbeliefWeights/ cumulativebeliefWeight * p_vals[pointer]
        if adjustedFDCR[pointer] > 1:
           adjustedFDCR[pointer] = 1.0  
        #step 2 - remember if (k) is rejected (1)..(k) are rejected - the FDR adjustemnt shoud reflect that
        if adjustedFDCR[pointer] > minAP:
            adjustedFDCR [pointer] = minAP
        else:
            minAP = adjustedFDCR[pointer]
        #update cumulative weight for the next step
        cumulativebeliefWeight = cumulativebeliefWeight - beliefWeights[pointer]
    return adjustedFDCR