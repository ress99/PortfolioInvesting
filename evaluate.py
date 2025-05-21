def return_risk(ind):

    prtf_return = ind.annualized_portfolio_return
    prtf_variance = ind.portfolio_risk()

    ind.fitness.values = prtf_return, prtf_variance

def pe_roe(ind):

    prtf_roe = ind.portfolio_roe()
    prtf_pe = ind.portfolio_pe()

    ind.fitness.values = prtf_roe, prtf_pe

def sharpe_var(ind):

    sharpe = ind.sharpe_ratio()
    var = ind.value_at_risk()

    ind.fitness.values = sharpe, var