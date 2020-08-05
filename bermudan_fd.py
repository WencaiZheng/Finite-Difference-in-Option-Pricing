from american_fd import *

class SOR_bermudan(SOR):

    def __init__(self, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon, ex_tvec):
        super().__init__(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon)
        self.ex_tvec = ex_tvec
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M1 = self.I + (1-self.theta)*self.dt*self.A
        self.M2 = self.I - self.theta*self.dt*self.A
    
    def _solve_(self):           
        w = self.alpha
        thedt = self.theta * self.dt
        payoff = self.grid[1:-1, -1]
        m = len(payoff)
        pastval = payoff.copy()

        idx = SOR_bermudan.get_idx( self.Tvec, self.ex_tvec )
        _, M_lower, M_upper = sla.lu(self.M2.toarray())  
        
        for j in reversed(np.arange(self.Nt)):
            z = self.M1.dot(pastval)

            z[0] += self.theta*self.l[0]*self.dt*self.grid[0, j] \
                 + (1-self.theta)*self.l[0]*self.dt*self.grid[0, j+1] 
            z[-1] += self.theta*self.u[-1]*self.dt*self.grid[-1, j] \
                  + (1-self.theta)*self.u[-1]*self.dt*self.grid[-1, j+1] 
            
            if j in idx: # update V = max(V, payoff)
                counter = 0
                noBreak = 1
                newval = pastval.copy()

                while noBreak:
                    counter += 1
                    oldval = newval.copy()
                    newval[0] = np.maximum( payoff[0], oldval[0] + w/(1-thedt*self.c[0]) \
                                           *( z[0] - (1-thedt*self.c[0])*oldval[0] \
                                             + thedt*self.u[0]*oldval[1]) )
                    for k in np.arange(1,m-1):
                        newval[k] = np.maximum( payoff[k], oldval[k] + w/(1-thedt*self.c[k]) \
                                               *( z[k] + thedt*self.l[k]*newval[k-1] \
                                                 - (1-thedt*self.c[k])*oldval[k] \
                                                 + thedt*self.u[k]*oldval[k+1]) )

                    newval[m-1] = np.maximum( payoff[m-1], oldval[m-1] + w/(1-thedt*self.c[m-1]) \
                                             *( z[m-1] + thedt*self.l[m-1]*newval[m-2] \
                                               - (1-thedt*self.c[m-1])*oldval[m-1]) )

                    noBreak = SOR_bermudan.trigger( oldval, newval, self.epsilon, counter, self.max_iter )

                pastval = newval.copy()
                
            else: # solve normal PDE            
                Ux = sla.solve_triangular( M_lower, z, lower=True )
                pastval = sla.solve_triangular( M_upper, Ux, lower=False )
            
            self.grid[1:-1, j] = pastval

    @staticmethod
    def get_tvec( tvec, ex_tvec ):
        all_tvec = np.unique( np.sort( np.concatenate([tvec,ex_tvec]) ) )
        return all_tvec

    @staticmethod
    def get_idx( tvec, ex_tvec ):
        all_tvec = SOR_bermudan.get_tvec( tvec, ex_tvec )
        return np.array( [np.where(all_tvec==t)[0][0] for t in ex_tvec] )


if __name__ == "__main__":
    
    # initial
    (S, K, r, q, T, sigma, option_type) = (50, 60, 0.03, 0.01, 1, 0.4, 'put')
    (Smin, Smax, Ns, Nt) = (0, 4*np.maximum(S,K), 200, 200)
    (theta, alpha, epsilon) = (0.5, 1.5, 1e-6)

    ex_tvec1 = np.arange(0.5, 1, 0.5)        # 每六个月可执行权利
    ex_tvec2 = np.arange(0.25, 1, 0.25)      # 每三个月可执行权利
    ex_tvec3 = np.arange(1/12, 1, 1/12)      # 每一个月可执行权利

    berm_opt1 = SOR_bermudan(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon, ex_tvec1)
    berm_opt2 = SOR_bermudan(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon, ex_tvec2)
    berm_opt3 = SOR_bermudan(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon, ex_tvec3)
    amer_opt = SOR(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon)

    opt = np.array( [berm_opt1.price(), berm_opt2.price(), berm_opt3.price(), amer_opt.price()] )

    print( f'The value of 6-month Bermudan {opt[0]}' )
    print( f'The value of 3-month Bermudan {opt[1]}' )
    print( f'The value of 1-month Bermudan {opt[2]}' )
    print( f'American option value is {opt[3]}' )