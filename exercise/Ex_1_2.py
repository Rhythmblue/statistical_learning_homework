from scipy.optimize import minimize

def main():
    x = (0, 1)
    f = lambda x: 10 - x[0] * x[0] - x[1] * x[1]
    cons = ({'type': 'ineq', 'fun': lambda x: x[1]-x[0]*x[0]},
            {'type': 'eq','fun': lambda x: x[0]+x[1]})
    res = minimize(fun=f, x0=x, method='SLSQP', constraints=cons)
    print(res)

if __name__ == '__main__':
    main()
