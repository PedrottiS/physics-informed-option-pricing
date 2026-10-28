#include "black_scholes.hpp"
#include "crank_nicolson.hpp"
#include "thomas.hpp"
#include <cmath>
#include <iostream>

using namespace pinn;

static int passed = 0, failed = 0;

#define CHECK(cond) do { \
    if (cond) { ++passed; } else { ++failed; \
        std::cerr << "FAIL: " #cond " @ line " << __LINE__ << "\n"; } \
} while (0)

#define CHECK_CLOSE(a, b, tol) do { \
    if (std::abs((a) - (b)) < (tol)) { ++passed; } else { ++failed; \
        std::cerr << "FAIL: |" << (a) << " - " << (b) << "| < " << (tol) \
                  << " @ line " << __LINE__ << "\n"; } \
} while (0)

static void test_thomas_identity() {
    const int n = 5;
    Vec a = Vec::Zero(n), b = Vec::Ones(n), c = Vec::Zero(n);
    Vec d(n); d << 1, 2, 3, 4, 5;
    const Vec x = ThomasSolver::solve_once(a, b, c, d);
    for (int i = 0; i < n; ++i) CHECK_CLOSE(x(i), d(i), 1e-14);
}

static void test_thomas_residual() {
    Vec a(3), b(3), c(3), d(3);
    a << 0, 1, 1;  b << 4, 4, 4;  c << 1, 1, 0;  d << 1, 2, 3;
    const Vec x = ThomasSolver::solve_once(a, b, c, d);
    CHECK_CLOSE(b(0)*x(0) + c(0)*x(1),              d(0), 1e-12);
    CHECK_CLOSE(a(1)*x(0) + b(1)*x(1) + c(1)*x(2),  d(1), 1e-12);
    CHECK_CLOSE(a(2)*x(1) + b(2)*x(2),              d(2), 1e-12);
}

static void test_thomas_reuse() {
    Vec a(3), b(3), c(3);
    a << 0, 1, 1;  b << 4, 4, 4;  c << 1, 1, 0;
    ThomasSolver ts(a, b, c);
    CHECK(ts.is_factored());
    Vec d1(3); d1 << 1, 2, 3;
    Vec d2(3); d2 << 3, 2, 1;
    const Vec x1 = ts.solve(d1), x2 = ts.solve(d2);
    const Vec r1 = ThomasSolver::solve_once(a, b, c, d1);
    const Vec r2 = ThomasSolver::solve_once(a, b, c, d2);
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE(x1(i), r1(i), 1e-12);
        CHECK_CLOSE(x2(i), r2(i), 1e-12);
    }
}

static void test_cn_matches_bs() {
    BSParams bs; GridConfig g; g.M = 800; g.Nt = 800; g.S0 = 100;
    CrankNicolson solver(bs, g);
    const double c   = CrankNicolson::price_at(solver.solve_european_call(), 100.0);
    const double ref = bs_call(100.0, bs.K, bs.T, bs.r, bs.sigma);
    CHECK_CLOSE(c, ref, 1e-3);
}

static void test_cn_boundaries() {
    BSParams bs; GridConfig g; g.M = 200; g.Nt = 200; g.S0 = 100;
    CrankNicolson solver(bs, g);
    const auto sol = solver.solve_european_call();
    CHECK_CLOSE(sol.V(0), 0.0, 1e-12);
    const double expected = sol.S(sol.S.size() - 1) - bs.K * std::exp(-bs.r * bs.T);
    CHECK_CLOSE(sol.V(sol.V.size() - 1), expected, 1e-10);
}

static void test_put_call_parity() {
    BSParams bs; GridConfig g; g.M = 400; g.Nt = 400; g.S0 = 100;
    CrankNicolson solver(bs, g);
    const double c = CrankNicolson::price_at(solver.solve_european_call(), 100.0);
    const double p = CrankNicolson::price_at(solver.solve_european_put(),  100.0);
    const double rhs = 100.0 - bs.K * std::exp(-bs.r * bs.T);
    CHECK_CLOSE(c - p, rhs, 1e-4);  // tolleranza più larga: errore FD
}

int main() {
    test_thomas_identity();
    test_thomas_residual();
    test_thomas_reuse();
    test_cn_matches_bs();
    test_cn_boundaries();
    test_put_call_parity();
    std::cout << "\n=== Unit tests ===\n"
              << "Passed: " << passed << "\n"
              << "Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}