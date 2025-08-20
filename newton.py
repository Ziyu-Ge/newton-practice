import numpy as np

def optimize(x0, f, tol=1e-6, max_iter=100, h=1e-5):
    """
    Univariate Newton's method for optimization using finite differences.
    """
    def fprime(x):
        return (f(x + h) - f(x - h)) / (2*h)
    
    def fsecond(x):
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    
    x = x0
    history = [x]
    
    for _ in range(max_iter):
        grad = fprime(x)
        hess = fsecond(x)
        
        if abs(hess) < 1e-12:
            print("Hessian is (near) zero, stopping.")
            break
        
        x_new = x - grad / hess
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            break
        
        x = x_new
    
    return x, history

