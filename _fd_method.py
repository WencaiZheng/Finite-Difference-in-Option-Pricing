from _option_pricing import *

class FiniteDifference(OptionPricingMethod):
    
    def __init__(self, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt):
        super().__init__(S, K, r, q, T, sigma, option_type)
        self.Smin = Smin
        self.Smax = Smax
        self.Ns = int(Ns)
        self.Nt = int(Nt)
        self.dS = (Smax-Smin)/Ns * 1.0
        self.dt = T/Nt*1.0
        self.Svec = np.linspace(Smin, Smax, self.Ns+1)
        self.Tvec = np.linspace(0, T, self.Nt+1)
        self.grid = np.zeros(shape=(self.Ns+1, self.Nt+1))
        
    def _set_terminal_condition_(self):
        self.grid[:, -1] = np.maximum(self.omega*(self.Svec - self.K), 0)
    
    def _set_boundary_condition_(self):
        tau = self.Tvec[-1] - self.Tvec;     
        DFq = np.exp(-self.q * tau)
        DFr = np.exp(-self.r * tau)
        # american option boundary condition should differ from Euro style
        self.grid[0,  :] = np.maximum(self.omega*(self.Svec[0]  - self.K), 0)
        self.grid[-1, :] = np.maximum(self.omega*(self.Svec[-1] - self.K), 0)        
        
    def _set_coefficient__(self):
        drift = (self.r-self.q)*self.Svec[1:-1]/self.dS
        diffusion_square = (self.sigma*self.Svec[1:-1]/self.dS)**2
        
        self.l = 0.5*(diffusion_square - drift)
        self.c = -diffusion_square - self.r
        self.u = 0.5*(diffusion_square + drift)
        
    def _solve_(self):
        pass
    
    def _interpolate_(self):
        tck = spi.splrep( self.Svec, self.grid[:,0], k=3 )
        return spi.splev( self.S, tck )
        #return np.interp(self.S, self.Svec, self.grid[:,0])
    
    def price(self):
        self._set_terminal_condition_()
        self._set_boundary_condition_()
        self._set_coefficient__()
        self._set_matrix_()
        self._solve_()
        return self._interpolate_()


class FullyExplicitEu(FiniteDifference):
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M = self.I + self.dt*self.A
                                        
    def _solve_(self):
        for j in reversed(np.arange(self.Nt)):
            U = self.M.dot(self.grid[1:-1, j+1])
            U[0] += self.l[0]*self.dt*self.grid[0, j+1] 
            U[-1] += self.u[-1]*self.dt*self.grid[-1, j+1] 
            self.grid[1:-1, j] = U


class FullyImplicitEu(FiniteDifference):

    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M = self.I - self.dt*self.A
    
    def _solve_(self):  
        _, M_lower, M_upper = sla.lu(self.M.toarray())

        for j in reversed(np.arange(self.Nt)):      
            U = self.grid[1:-1, j+1].copy()
            U[0] += self.l[0]*self.dt*self.grid[0, j] 
            U[-1] += self.u[-1]*self.dt*self.grid[-1, j] 
            Ux = sla.solve_triangular( M_lower, U, lower=True )
            self.grid[1:-1, j] = sla.solve_triangular( M_upper, Ux, lower=False )


class CrankNicolsonEu(FiniteDifference):

    theta = 0.5
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M1 = self.I + (1-self.theta)*self.dt*self.A
        self.M2 = self.I - self.theta*self.dt*self.A
    
    def _solve_(self):           
        _, M_lower, M_upper = sla.lu(self.M2.toarray())        
        for j in reversed(np.arange(self.Nt)):
            
            U = self.M1.dot(self.grid[1:-1, j+1])
            
            U[0] += self.theta*self.l[0]*self.dt*self.grid[0, j] \
                 + (1-self.theta)*self.l[0]*self.dt*self.grid[0, j+1] 
            U[-1] += self.theta*self.u[-1]*self.dt*self.grid[-1, j] \
                  + (1-self.theta)*self.u[-1]*self.dt*self.grid[-1, j+1] 
            
            Ux = sla.solve_triangular( M_lower, U, lower=True )
            self.grid[1:-1, j] = sla.solve_triangular( M_upper, Ux, lower=False )
