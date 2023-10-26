def evaluate(ind):

    prtf_return = ind.portfolio_return()
    prtf_variance = ind.portfolio_variance()

    ind.fitness.values = prtf_return, prtf_variance

def pe_roe(ind):

    prtf_roe = ind.portfolio_roe()
    prtf_pe = ind.portfolio_pe()

    ind.fitness.values = prtf_roe, prtf_pe