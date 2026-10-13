// Educational tauRAMD-style stochastic escape model, not atomistic MD.
// Inspired by Kokh et al. (2018), JCTC 14, 3859, and subsequent RAMD theory.
// Dimensionless units. Center-of-mass (COM) is a single effective particle.
// A bounded attractive well represents a receptor; white thermal noise and a
// fixed-magnitude, piecewise-constant random force accelerate escape.
// The direction is reassigned whenever progress in a check interval is small.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

struct Vec3 {
    double x, y, z;
    Vec3 operator+(Vec3 b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(Vec3 b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(double a) const { return {a*x, a*y, a*z}; }
};

double norm(Vec3 a) { return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }

Vec3 uniform_unit_vector(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    constexpr double pi = 3.14159265358979323846;
    const double z = 2.0*uniform(rng)-1.0;
    const double phi = 2.0*pi*uniform(rng);
    const double radial = std::sqrt(std::max(0.0, 1.0-z*z));
    return {radial*std::cos(phi), radial*std::sin(phi), z};
}

// Smooth finite well: U(r) = -depth * exp(-r^2/(2*width^2)).
// It exerts inward force: -grad(U).
Vec3 well_force(Vec3 position, double depth, double width) {
    const double r2 = position.x*position.x + position.y*position.y + position.z*position.z;
    const double factor = -depth/(width*width) * std::exp(-r2/(2.0*width*width));
    return position*factor;
}

struct Config {
    int trajectories = 24;
    int max_steps = 200000;
    int check_every = 80;
    int save_every = 40;
    double dt = 0.002;
    double mobility = 1.0;
    double kbt = 1.0;
    double well_depth = 7.0;
    double well_width = 1.0;
    double random_force = 7.0;
    double min_progress = 0.12;
    double escape_radius = 3.0;
    std::uint64_t seed = 2026;
};

int main(int argc, char** argv) {
    try {
        Config c;
        // Optional first command-line argument: force magnitude; second: seed.
        if (argc > 1) c.random_force = std::stod(argv[1]);
        if (argc > 2) c.seed = static_cast<std::uint64_t>(std::stoull(argv[2]));
        if (c.random_force < 0 || c.dt <= 0 || c.mobility <= 0 || c.kbt <= 0 ||
            c.trajectories <= 0 || c.max_steps <= 0 || c.check_every <= 0 ||
            c.escape_radius <= 0 || c.save_every <= 0 || c.min_progress < 0 ||
            c.well_width <= 0 || c.well_depth < 0) {
            throw std::invalid_argument("Invalid simulation parameters");
        }
        std::ofstream trajectory_file("tau_ramd_trajectories.csv");
        std::ofstream summary_file("tau_ramd_escapes.csv");
        if (!trajectory_file || !summary_file) throw std::runtime_error("Could not open output files");
        trajectory_file << "run,step,time,x,y,z,r,direction_x,direction_y,direction_z,changed_direction\n";
        summary_file << "run,escaped,escape_time,force_magnitude,seed,direction_changes\n";
        trajectory_file << std::setprecision(10);
        summary_file << std::setprecision(10);
        const double noise_scale = std::sqrt(2.0*c.mobility*c.kbt*c.dt);
        int escaped_count=0;
        double escaped_time_sum=0.0;
        for (int run=0; run<c.trajectories; ++run) {
            std::mt19937_64 rng(c.seed+static_cast<std::uint64_t>(run)*1000003ULL);
            std::normal_distribution<double> normal(0.0,1.0);
            Vec3 position{0,0,0};
            Vec3 reference_position=position;
            Vec3 direction=uniform_unit_vector(rng);
            int changes=0;
            bool escaped=false;
            double escape_time=std::numeric_limits<double>::quiet_NaN();
            for (int step=1; step<=c.max_steps; ++step) {
                const Vec3 physical_force=well_force(position,c.well_depth,c.well_width);
                const Vec3 applied_force=physical_force+direction*c.random_force;
                const Vec3 noise{normal(rng),normal(rng),normal(rng)};
                position=position + applied_force*(c.mobility*c.dt) + noise*noise_scale;
                const double r=norm(position);
                int changed=0;
                if (r >= c.escape_radius) {
                    escaped=true;
                    escape_time=step*c.dt;
                } else if (step % c.check_every == 0) {
                    // Net COM displacement over check interval, not a single step.
                    const double displacement=norm(position-reference_position);
                    if (displacement < c.min_progress) {
                        direction=uniform_unit_vector(rng);
                        changed=1;
                        ++changes;
                    }
                    reference_position=position;
                }
                if (step == 1 || step % c.save_every == 0 || escaped) {
                    trajectory_file << run << ',' << step << ',' << step*c.dt << ','
                                    << position.x << ',' << position.y << ',' << position.z << ','
                                    << r << ',' << direction.x << ',' << direction.y << ','
                                    << direction.z << ',' << changed << '\n';
                }
                if (escaped) break;
            }
            if (escaped) { ++escaped_count; escaped_time_sum+=escape_time; }
            summary_file << run << ',' << (escaped ? 1 : 0) << ',';
            if (escaped) summary_file << escape_time;
            summary_file << ',' << c.random_force << ',' << (c.seed+static_cast<std::uint64_t>(run)*1000003ULL)
                         << ',' << changes << '\n';
        }
        std::cout << "tauRAMD-style toy model (dimensionless; no atomistic receptor)\n"
                  << "force=" << c.random_force << ", escaped=" << escaped_count << '/' << c.trajectories << '\n';
        if (escaped_count) std::cout << "Mean escape time among escaped runs only="
                                     << escaped_time_sum/escaped_count << '\n';
        if (escaped_count < c.trajectories) std::cout << "WARNING: some runs are censored at max_steps; escaped-only mean is biased.\n";
        std::cout << "Saved tau_ramd_trajectories.csv and tau_ramd_escapes.csv\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
