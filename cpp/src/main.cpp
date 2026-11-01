#include "black_scholes.hpp"
#include "crank_nicolson.hpp"
#include "pinn_onnx.hpp" // <-- Aggiunto include per la PINN
#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <vector>

using namespace pinn;

int main() {
    BSParams bs;
    const double S0  = 100.0;
    const double ref = bs_call(S0, bs.K, bs.T, bs.r, bs.sigma);

    std::cout << std::setprecision(10);
    std::cout << "=== M2: Crank-Nicolson vs Black-Scholes (European call) ===\n";
    std::cout << "Params: S0=" << S0 << " K=" << bs.K << " T=" << bs.T
              << " r=" << bs.r << " sigma=" << bs.sigma << "\n";
    std::cout << "BS analytic: " << ref << "\n\n";

    std::cout << std::setw(6)  << "M"
              << std::setw(6)  << "Nt"
              << std::setw(18) << "CN"
              << std::setw(14) << "|err|" << "\n";
    std::cout << std::string(44, '-') << "\n";

    for (int N : {50, 100, 200, 400, 800}) {
        GridConfig g; g.M = N; g.Nt = N; g.S0 = S0;
        CrankNicolson solver(bs, g);
        const auto sol = solver.solve_european_call();
        const double v = CrankNicolson::price_at(sol, S0);
        std::cout << std::setw(6)  << N
                  << std::setw(6)  << N
                  << std::setw(18) << std::fixed << std::setprecision(8) << v
                  << std::setw(14) << std::scientific << std::setprecision(3)
                  << std::abs(v - ref) << "\n";
    }

    // Arbitrage bounds on call
    std::cout << "\n=== Arbitrage bounds ===\n";
    GridConfig g; g.M = 400; g.Nt = 400; g.S0 = S0;
    CrankNicolson solver(bs, g);
    const auto sol = solver.solve_european_call();
    const double c     = CrankNicolson::price_at(sol, S0);
    const double lower = std::max(S0 - bs.K * std::exp(-bs.r * bs.T), 0.0);
    std::cout << "lower = " << lower << "\n"
              << "C_FD  = " << c     << "\n"
              << "upper = " << S0    << "\n"
              << "bounds: "
              << ((c >= lower - 1e-10 && c <= S0 + 1e-10) ? "OK" : "FAILED")
              << "\n";

    // Monotonicity in S0
    std::cout << "\n=== Monotonicity in S0 ===\n";
    bool monotone = true;
    double prev = -1.0;
    for (double s : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        const double cs = CrankNicolson::price_at(sol, s);
        if (cs < prev) monotone = false;
        std::cout << "  C(S0=" << std::setw(3) << s << ") = " << cs << "\n";
        prev = cs;
    }
    std::cout << "monotone: " << (monotone ? "OK" : "FAILED") << "\n";

    // Put-call parity
    std::cout << "\n=== Put-call parity ===\n";
    const double c_fd = CrankNicolson::price_at(solver.solve_european_call(), S0);
    const double p_fd = CrankNicolson::price_at(solver.solve_european_put(),  S0);
    const double lhs  = c_fd - p_fd;
    const double rhs  = S0 - bs.K * std::exp(-bs.r * bs.T);
    std::cout << "C - P (FD)      = " << lhs << "\n";
    std::cout << "S - K e^{-rT}   = " << rhs << "\n";
    std::cout << "|diff|          = " << std::abs(lhs - rhs) << "\n";

    // =========================================================================
    // M7: PINN ONNX TEST & LATENCY BENCHMARK
    // =========================================================================
    std::cout << "\n=== M7: PINN ONNX Inference Test ===\n";
    try {
        // Assicurati che il percorso del modello sia corretto!
        PinnOnnx pinn_model("../models/pinn_bs.onnx");
        
        float tau = static_cast<float>(bs.T); // tau = T (cioè oggi)
        float S   = static_cast<float>(S0);
        
        float pinn_price = pinn_model.predict(tau, S);
        
        std::cout << "Input: tau=" << tau << ", S=" << S << "\n";
        std::cout << "PINN price:      " << pinn_price << "\n";
        std::cout << "BS analytic:     " << ref << "\n";
        std::cout << "Absolute error:  " << std::abs(pinn_price - ref) << "\n";

        // Latenza PINN (facciamo 1000 run per fare una media)
        const int N_runs = 1000;
        auto start_pinn = std::chrono::high_resolution_clock::now();
        float sum = 0.0f;
        for (int i = 0; i < N_runs; ++i) {
            sum += pinn_model.predict(tau, S);
        }
        auto end_pinn = std::chrono::high_resolution_clock::now();
        auto duration_pinn = std::chrono::duration_cast<std::chrono::microseconds>(end_pinn - start_pinn).count();
        double avg_pinn_us = static_cast<double>(duration_pinn) / N_runs;

        // Latenza Crank-Nicolson (facciamo 10 run, è più lento)
        const int N_runs_cn = 10;
        auto start_cn = std::chrono::high_resolution_clock::now();
        double sum_cn = 0.0;
        for (int i = 0; i < N_runs_cn; ++i) {
            sum_cn += CrankNicolson::price_at(solver.solve_european_call(), S0);
        }
        auto end_cn = std::chrono::high_resolution_clock::now();
        auto duration_cn = std::chrono::duration_cast<std::chrono::microseconds>(end_cn - start_cn).count();
        double avg_cn_us = static_cast<double>(duration_cn) / N_runs_cn;

        std::cout << "\n--- Latency Benchmark ---\n";
        std::cout << "PINN (C++/ONNX): " << avg_pinn_us << " us/price\n";
        std::cout << "Crank-Nicolson:  " << avg_cn_us << " us/price\n";
        std::cout << "Speedup:         " << (avg_cn_us / avg_pinn_us) << "x\n";

    } catch (const std::exception& e) {
        std::cerr << "ERRORE ONNX: " << e.what() << "\n";
        std::cerr << "Controlla che il file .onnx e la cartella onnxruntime siano nel posto giusto.\n";
    }

    return 0;
}