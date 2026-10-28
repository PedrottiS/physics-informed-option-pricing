#pragma once
#include <cmath>

namespace pinn {

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

inline double bs_call(double S, double K, double T, double r, double sigma) {
    const double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
    const double d2 = d1 - sigma*std::sqrt(T);
    return S*norm_cdf(d1) - K*std::exp(-r*T)*norm_cdf(d2);
}

inline double bs_put(double S, double K, double T, double r, double sigma) {
    const double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
    const double d2 = d1 - sigma*std::sqrt(T);
    return K*std::exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1);
}

} // namespace pinn