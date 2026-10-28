#pragma once
#include <Eigen/Dense>
#include <cassert>
#include "types.hpp"

namespace pinn {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

class ThomasSolver {
public:
    ThomasSolver() = default;

    ThomasSolver(const Vec& a, const Vec& b, const Vec& c) {
        factorize(a, b, c);
    }

    static Vec solve_once(const Vec& a, const Vec& b, const Vec& c, const Vec& d) {
        ThomasSolver ts(a, b, c);
        return ts.solve(d);
    }

    void factorize(const Vec& a, const Vec& b, const Vec& c) {
        const int n = static_cast<int>(b.size());
        assert(static_cast<int>(a.size()) == n
            && static_cast<int>(c.size()) == n
            && n > 0 && "wrong dimensions");

        cp_.resize(n);
        inv_bp_.resize(n);
        a_over_bp_.resize(n);

        inv_bp_(0)    = 1.0 / b(0);
        a_over_bp_(0) = 0.0;
        cp_(0)        = (n > 1) ? c(0) * inv_bp_(0) : 0.0;

        for (int i = 1; i < n; ++i) {
            const double bp = b(i) - a(i) * cp_(i - 1);
            inv_bp_(i)      = 1.0 / bp;
            a_over_bp_(i)   = a(i) * inv_bp_(i);
            cp_(i)          = (i < n - 1) ? c(i) * inv_bp_(i) : 0.0;
        }
        factored_ = true;
    }

    Vec solve(const Vec& d) const {
        assert(factored_ && "call factorize() first");
        const int n = static_cast<int>(d.size());
        assert(n == static_cast<int>(inv_bp_.size()) && "RHS dimension mismatch");

        Vec dp(n);
        dp(0) = d(0) * inv_bp_(0);
        for (int i = 1; i < n; ++i)
            dp(i) = d(i) * inv_bp_(i) - a_over_bp_(i) * dp(i - 1);

        Vec x(n);
        x(n - 1) = dp(n - 1);
        for (int i = n - 2; i >= 0; --i)
            x(i) = dp(i) - cp_(i) * x(i + 1);

        return x;
    }

    bool is_factored() const { return factored_; }
    int  size()       const { return static_cast<int>(inv_bp_.size()); }

private:
    Vec  cp_;
    Vec  inv_bp_;
    Vec  a_over_bp_;
    bool factored_ = false;
};

} // namespace pinn