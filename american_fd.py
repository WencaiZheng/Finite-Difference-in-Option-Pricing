from _fd_method import *
# iterate method
class SOR(FiniteDifference):

    def __init__(self, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon):
        super().__init__(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt)
        self.theta = theta
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = 10*Nt
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M1 = self.I + (1-self.theta)*self.dt*self.A
    
    def _solve_(self):           
        w = self.alpha
        thedt = self.theta * self.dt
        payoff = self.grid[1:-1, -1]
        m = len(payoff)
        pastval = payoff.copy()
        
        for j in reversed(np.arange(self.Nt)):
            counter = 0
            noBreak = 1
            newval = pastval.copy()
            
            z = self.M1.dot(pastval)
            
            z[0] += self.theta*self.l[0]*self.dt*self.grid[0, j] \
                 + (1-self.theta)*self.l[0]*self.dt*self.grid[0, j+1] 
            z[-1] += self.theta*self.u[-1]*self.dt*self.grid[-1, j] \
                  + (1-self.theta)*self.u[-1]*self.dt*self.grid[-1, j+1] 
            
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
        
                noBreak = SOR.trigger( oldval, newval, self.epsilon, counter, self.max_iter )
                
            pastval = newval.copy()
            self.grid[1:-1, j] = pastval
      
    @staticmethod
    def trigger( oldval, newval, tol, counter, maxIteration ):
        noBreak = 1
        if np.max( np.abs(newval-oldval)/np.maximum(1,np.abs(newval)) ) <= tol:
            noBreak = 0
        elif counter > maxIteration:
            print('The result may not converge.')
            noBreak = 0
        return noBreak

# penalty method
class PM(FiniteDifference):

    def __init__(self, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, lbd, epsilon):
        super().__init__(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt)
        self.theta = theta
        self.lbd = lbd
        self.epsilon = epsilon
        self.max_iter = 10*Nt
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1], format='csc')
        self.I = sp.eye(self.Ns-1)
    
    def _solve_(self):           
        (theta, dt) = (self.theta, self.dt)
        payoff = self.grid[1:-1, -1]
        pastval = payoff.copy()
        G = payoff.copy()
        
        for j in reversed(np.arange(self.Nt)):
            counter = 0
            noBreak = 1
            newval = pastval.copy()
            
            while noBreak:
                counter += 1
                oldval = newval.copy()
                D = sp.diags( (G > (1-theta)*pastval + theta*newval).astype(int), format='csc' )
                z = (self.I + (1-theta)*dt*(self.A - self.lbd*D))*pastval + dt*self.lbd*D*G
                
                z[0] += theta*self.l[0]*dt*self.grid[0, j] \
                 + (1-theta)*self.l[0]*dt*self.grid[0, j+1] 
                z[-1] += theta*self.u[-1]*dt*self.grid[-1, j] \
                  + (1-theta)*self.u[-1]*dt*self.grid[-1, j+1] 
                                
                M = self.I - theta*dt*(self.A - self.lbd*D)
                newval = spsolve(M,z)
        
                noBreak = PM.trigger( oldval, newval, self.epsilon, counter, self.max_iter )
            
            pastval = newval.copy()
            self.grid[1:-1, j] = pastval
    
    @staticmethod
    def trigger( oldval, newval, tol, counter, maxIteration ):
        noBreak = 1
        if np.max( np.abs(newval-oldval)/np.maximum(1,np.abs(newval)) ) <= tol:
            noBreak = 0
        elif counter > maxIteration:
            print('结果可能不收敛。')
            noBreak = 0
        return noBreak


if __name__ == "__main__":
    
    # initial
    (S, K, r, q, T, sigma, option_type) = (50, 60, 0.03, 0.01, 1, 0.4, 'put')
    (Smin, Smax, Ns, Nt) = (0, 4*np.maximum(S,K), 200, 200)
    (theta, alpha, epsilon) = (0.5, 1.5, 1e-6)
    amer_opt = SOR(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt, theta, alpha, epsilon)
    print(amer_opt.price())