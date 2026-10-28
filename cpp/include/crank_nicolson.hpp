#pragma once
#include "types.hpp"
#include "thomas.hpp"
#include <algorithm>
#include <cmath>

namespace pinn {

struct BSParams {
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
};

struct GridConfig {
    int    M = 400;    // intervals in S
    int    Nt = 400;    // steps in tau
    double Smax_factor = 5.0;    // Smax = factor * max(S0, K)
    double S0 = 100.0; 
};

struct Solution {
    Vec    S;       
    Vec    V;       
    double dS = 0.0;
};

class CrankNicolson {
public:
    CrankNicolson(const BSParams& bs, const GridConfig& grid) : bs_(bs), grid_(grid) {
        Smax_ = grid_.Smax_factor * std::max(grid_.S0, bs_.K);
        M_ = grid_.M;
        Nt_ = grid_.Nt;
        dS_ = Smax_ / M_;
        dtau_ = bs_.T / Nt_;
        build_operator();
        build_lhs();
    }

    Solution solve_european_call() const { return solve_european(true);  }
    Solution solve_european_put()  const { return solve_european(false); }

    static double price_at(const Solution& sol, double S0) {
        const double pos = S0 / sol.dS;
        const int    i0 = static_cast<int>(std::floor(pos));
        const double w = pos - i0;
        if (i0 < 0)                    return sol.V(0);
        if (i0 >= static_cast<int>(sol.V.size()) - 1)
                                       return sol.V(sol.V.size() - 1);
        return (1.0 - w) * sol.V(i0) + w * sol.V(i0 + 1);
    }

private:
    // L = 0.5*sigma^2*S^2*d^2/dS^2 + r*S*d/dS - r
    // on internal nodes i = 1..M-1:  L V_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1}
    void build_operator() {
        const int n = M_ - 1;
        a_.resize(n); b_.resize(n); c_.resize(n);
        const double s2 = bs_.sigma * bs_.sigma;
        const double inv_dS2 = 1.0 / (dS_ * dS_);
        for (int idx = 0; idx < n; ++idx) {
            const int i = idx + 1;
            const double Si = i * dS_;
            const double s2S2 = s2 * Si * Si;
            a_(idx) =  0.5 * s2S2 * inv_dS2 - 0.5 * bs_.r * Si / dS_;
            b_(idx) = -s2S2 * inv_dS2 - bs_.r;
            c_(idx) =  0.5 * s2S2 * inv_dS2 + 0.5 * bs_.r * Si / dS_;
        }
    }

    // LHS = I - 0.5*dtau*L  (fattorizzata una sola volta)
    void build_lhs() {
        const int n = M_ - 1;
        Vec la(n), lb(n), lc(n);
        for (int idx = 0; idx < n; ++idx) {
            la(idx) = -0.5 * dtau_ * a_(idx);
            lb(idx) =  1.0 - 0.5 * dtau_ * b_(idx);
            lc(idx) = -0.5 * dtau_ * c_(idx);
        }
        lhs_ = ThomasSolver(la, lb, lc);
    }

    Solution solve_european(bool is_call) const {
        const int n = M_ - 1;

        // Condizione iniziale tau = 0: payoff
        Vec V(n);
        for (int idx = 0; idx < n; ++idx) {
            const double S = (idx + 1) * dS_;
            V(idx) = is_call ? std::max(S - bs_.K, 0.0)
                             : std::max(bs_.K - S, 0.0);
        }

        // Marcia in tau
        for (int step = 1; step <= Nt_; ++step) {
            const double tau_old = (step - 1) * dtau_;
            const double tau_new = step * dtau_;

            double V0_old, V0_new, VM_old, VM_new;
            if (is_call) {
                V0_old = 0.0; V0_new = 0.0;
                VM_old = Smax_ - bs_.K * std::exp(-bs_.r * tau_old);
                VM_new = Smax_ - bs_.K * std::exp(-bs_.r * tau_new);
            } else {
                V0_old = bs_.K * std::exp(-bs_.r * tau_old);
                V0_new = bs_.K * std::exp(-bs_.r * tau_new);
                VM_old = 0.0; VM_new = 0.0;
            }

            // RHS = (I + 0.5*dtau*L) V^n
            Vec rhs(n);
            for (int idx = 0; idx < n; ++idx) {
                const double left  = (idx > 0)     ? 0.5 * dtau_ * a_(idx) * V(idx - 1) : 0.0;
                const double right = (idx < n - 1) ? 0.5 * dtau_ * c_(idx) * V(idx + 1) : 0.0;
                rhs(idx) = left + (1.0 + 0.5 * dtau_ * b_(idx)) * V(idx) + right;
            }
            // BC: media old/new (punto critico dello schema CN)
            rhs(0) += 0.5 * dtau_ * a_(0)     * (V0_old + V0_new);
            rhs(n - 1) += 0.5 * dtau_ * c_(n - 1) * (VM_old + VM_new);

            V = lhs_.solve(rhs);
        }

        // Ricostruisci soluzione completa (M+1 nodi incluso bordo)
        Solution sol;
        sol.dS = dS_;
        sol.S.resize(M_ + 1);
        sol.V.resize(M_ + 1);
        for (int i = 0; i <= M_; ++i) sol.S(i) = i * dS_;
        sol.V(0) = is_call ? 0.0 : bs_.K * std::exp(-bs_.r * bs_.T);
        for (int i = 1; i < M_; ++i) sol.V(i) = V(i - 1);
        sol.V(M_) = is_call ? Smax_ - bs_.K * std::exp(-bs_.r * bs_.T) : 0.0;
        return sol;
    }

    BSParams bs_;
    GridConfig grid_;
    double Smax_ = 0.0;
    double dS_   = 0.0;
    double dtau_ = 0.0;
    int M_ = 0;
    int Nt_ = 0;
    Vec a_, b_, c_;
    ThomasSolver lhs_;
};

} // namespace pinn