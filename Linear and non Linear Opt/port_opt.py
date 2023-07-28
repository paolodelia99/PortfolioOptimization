import numpy as np
import scipy.optimize as sco

def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

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
                        'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq',
                        'fun': lambda x: x})
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