from _fd_method import *

if __name__=='__main__':
    (S0, K, r, q, T, sigma, omega) = (50, 60, 0.03, 0.01, 1, 0.4, -1)
    (S, K, r, q, T, sigma, option_type) = (50, 60, 0.03, 0.01, 1, 0.4, 'put')

    (Smin, Smax, Ns, Nt) = (0, 4*np.maximum(S,K), 100, 100)

    # explicit method should have small time steps and large stock price
    option = FullyExplicitEu(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns=20, Nt=100)
    print(option.price())
    # implicit method always converge
    option2 = FullyImplicitEu(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt)
    print(option2.price())

    # so as the Crank Nicolson
    option3 = CrankNicolsonEu(S, K, r, q, T, sigma, option_type, Smin, Smax, 100, 100)
    print(option3.price())

    print(blackscholes(S0, K, r, q, T, sigma, omega))