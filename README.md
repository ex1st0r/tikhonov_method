# Tikhonov regularization
Tikhonov regularization method to solve the Fredholm integral equation of the first kind.
Script was written on Python 2.7.
<h2>Using:</h2>
Import class TikhonovMethod from file tikhonov_method.py or modify contents of this file.
To run the solution, write:
<pre>
fredholm = TikhonovMethod() #Class initialization
fredholm.kernel_func = lambda x,s: 1/(1+100*(x-s)**2) #Kernel
fredholm.z_func = lambda s: exp(-(s-0.5)**2/0.06) #Exact solution [z]
fredholm.solve(a = 0, b = 1, c = 0, d = 1, n = 41, m = 41, delta = 10**-8, h = 10**-10) #Start process
</pre>
