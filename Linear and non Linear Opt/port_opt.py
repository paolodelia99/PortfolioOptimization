import numpy as np
import scipy.optimize as sco

def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def neg_portf_rtn(w, avg_rtns):
    return -get_portf_rtn(w, avg_rtns)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def get_utility_value(w, avg_rtns, cov_mat, risk_aversion):
    portf_rtn = get_portf_rtn(w, avg_rtns)
    portf_vol = get_portf_vol(w, avg_rtns, cov_mat)
    return portf_rtn - 0.5 * risk_aversion * portf_vol

def neg_utility_value(w, avg_rtns, cov_mat, risk_aversion):
    return -get_utility_value(w, avg_rtns, cov_mat, risk_aversion)

def get_optimal_portfolio(f_obj, constraints, args, bounds):
    n_assets = len(args[0])
    initial_guess = n_assets * [1. / n_assets, ]
    return sco.minimize(f_obj,
                        x0=initial_guess,
                        args=args,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):

    efficient_portfolios = []

    n_assets = len(avg_rtns)
    args = (avg_rtns, cov_mat)
    bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]

    for ret in rtns_range:
        constraints = ({'type': 'eq',
                        'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret},
                       {'type': 'eq',
                        'fun': lambda x: np.sum(x) - 1})
        efficient_portfolio = get_optimal_portfolio(f_obj=get_portf_vol,
                                           args=args,
                                           constraints=constraints,
                                           bounds=bounds)
        efficient_portfolios.append(efficient_portfolio)

    return efficient_portfolios

def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    return -portf_sharpe_ratio