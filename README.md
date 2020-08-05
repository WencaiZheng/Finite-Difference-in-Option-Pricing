# Finite-Difference-in-Option-Pricing

Use Finite Difference method to price European, American and Bermudan options.

## Major steps

1. solution domain
2. grid construction
3. terminal and boundary condition
4. spatial and time discretization
5. finite difference scheme

## European PDE

PDE

<img src="example\europde.png" style="zoom: 50%;" />

The finite difference scheme

<img src="example\europdefd.png" style="zoom: 50%;" />

## American PDE

Delta hedge portfolio inequality, if execution timing is wrong, the portfolio value would be less:

<img src="example\12.png" style="zoom: 42%;" />

The American option inequality:

<img src="example\amerineq.png" style="zoom:55%;" />

For call option, w=1, for put, w =-1:

When V > w(S-K), PDE becomes European style:

<img src="example\21.png" style="zoom: 50%;" />

When V = w(S-K), PDE is:

<img src="example\22.png" style="zoom:55%;" />

The we have a simple form:

<img src="example\11.png" style="zoom:43%;" />

There are two ways: 

### Iteration Method

#### Jacobi

#### Gauss-Seidel, GS

#### successive over-relaxation, SOR

![](example\iteration.png)

### Penalty Method

American option PDE can be rewrite as a linear complementarity (LCP) problem as below.

<img src="example\amerlcp.png" style="zoom: 45%;" />

Where

<img src="example\23.png" style="zoom: 50%;" />

## Bermudan PDE

![](example\9.png)