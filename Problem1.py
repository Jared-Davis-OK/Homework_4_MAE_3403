#region problem statement
r'''
	(50 pts) Create a Python program that simulates an industrial-scale gravel production process where crushed rocks
	are sieved through a pair of screens:  the first screen is a large aperture screen that excludes rocks above a
	certain size and the second screen has a smaller aperture.  The product is the fraction of rocks from between the screens.

Assumptions:
	While the actual gavel is not spherical, we will assume that the rocks are spherical.
	Prior to sieving, the gravel follows a log-normal distribution (i.e., loge(D) is N(μ,σ)), where D is the rock
	diameter, μ=mean of ln(D) and σ= standard deviation of ln(D).  After sieving, the log-normal distribution is now
	truncated to have a maximum (Dmax) and minimum size (Dmin) imposed by the aperture size of the screens.

Your program should solicit input from the user (with suggested default values) μ, σ, Dmax and Dmin.  It should then
produce 11 samples of N=100 rocks randomly selected from the truncated log-normal distribution and report to the user
through the cli the sample mean (D̅) and variance (S2) of each sample as well as the mean and variance of the sampling mean.

Note:
The standard log-normal probability density function (PDF) is normalized over (0,∞) by:
f\left(D\right)=\frac{1}{D\sigma\sqrt{2\pi}}e^{-\frac{\left(ln\left(D\right)-\mu\right)^2}{2\sigma^2}};\int_{0}^{\infty}f\left(D\right)dD=1

And the normalized truncated log-normal PDF is given by:
f_{trunc}\left(D\right)=\frac{f\left(D\right)}{F\left(D_{max}\right)-F\left(D_{min}\right)}

Your grade will be based on your efficient use of imports of the allowed modules, use of functions and function calls,
use of lists and list comprehensions, your clarity in your docstrings and comments and your overall approach to the
problem.  Clearly state your assumptions in the docstring of the main function.
'''
#endregion

#region imports
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
#endregion


#region function definitions
#region probability calculations

"""
Lognormal and Truncated Lognormal Simulation
Uses quad() for integration and fsolve() for inverse CDF
"""


def LogNormalPDF(D, mu, sigma):
    """
    Computes the lognormal probability density function.
    :param D: rock diameter
    :param mu: mean of ln(D)
    :param sigma: standard deviation of ln(D)
    :return: PDF value
    """
    if D <= 0:
        return 0.0

    return (1 / (D * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-((np.log(D) - mu) ** 2) / (2 * sigma ** 2))


def LogNormalCDF(D, mu, sigma):
    """
    Computes the lognormal CDF using quad integration.
    :return: CDF value at D
    """
    value, _ = quad(LogNormalPDF, 0, D, args=(mu, sigma))
    return value


def TruncatedPDF(D, mu, sigma, Dmin, Dmax):
    """
    Computes truncated lognormal PDF.
    """
    Fmin = LogNormalCDF(Dmin, mu, sigma)
    Fmax = LogNormalCDF(Dmax, mu, sigma)

    if D < Dmin or D > Dmax:
        return 0.0

    return LogNormalPDF(D, mu, sigma) / (Fmax - Fmin)


def InverseTruncatedCDF(u, mu, sigma, Dmin, Dmax):

    Fmin = LogNormalCDF(Dmin, mu, sigma)
    Fmax = LogNormalCDF(Dmax, mu, sigma)

    def equation(D):
        D = D[0]   # extract scalar
        return LogNormalCDF(D, mu, sigma) - (Fmin + u * (Fmax - Fmin))

    Dguess = (Dmin + Dmax) / 2
    solution = fsolve(equation, Dguess)

    return solution[0]


#endregion
#endregion


#region main function
def main():
    """
    Main simulation driver
    """

    # ----- User Inputs -----
    mu = float(input("Mean of ln(D)? (0.693): ") or 0.693)
    sigma = float(input("Std dev of ln(D)? (1.000): ") or 1.000)
    Dmax = float(input("Large aperture Dmax? (1.000): ") or 1.000)
    Dmin = float(input("Small aperture Dmin? (0.375): ") or 0.375)

    nSamples = 11
    N = 100

    print("\nGenerating 11 samples of 100 rocks each...\n")

    sample_means = []
    sample_variances = []

    # ----- Sampling Loop -----
    for i in range(nSamples):

        sample = []

        for _ in range(N):
            u = np.random.uniform(0, 1)
            D = InverseTruncatedCDF(u, mu, sigma, Dmin, Dmax)
            sample.append(D)

        sample = np.array(sample)

        mean_i = np.mean(sample)
        var_i = np.var(sample, ddof=1)

        sample_means.append(mean_i)
        sample_variances.append(var_i)

        print(f"Sample {i+1}:")
        print(f"   Mean = {mean_i:.5f}")
        print(f"   Variance = {var_i:.5f}\n")

    # ----- Sampling Mean Statistics -----
    mean_of_means = np.mean(sample_means)
    var_of_means = np.var(sample_means, ddof=1)

    print("Overall Statistics of Sampling Means:")
    print(f"   Mean of Means = {mean_of_means:.5f}")
    print(f"   Variance of Means = {var_of_means:.5f}")


#endregion


#region function calls
if __name__ == "__main__":
    main()
#endregion
