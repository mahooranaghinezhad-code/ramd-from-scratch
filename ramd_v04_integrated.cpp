#include <iostream>
#include <cmath>
using namespace std;

int main() {

    // -----------------------------
    // 1. Define a simple 3-atom ligand
    // -----------------------------
    double mass1 = 12.0;
    double mass2 = 16.0;
    double mass3 = 1.0;

    double x1 = 1.0;
    double x2 = 3.0;
    double x3 = 5.0;

    double totalMass = mass1 + mass2 + mass3;

    // -----------------------------
    // 2. Calculate the initial COM
    // -----------------------------
    double oldCOM =
        (mass1 * x1 + mass2 * x2 + mass3 * x3)
        / totalMass;

    // -----------------------------
    // 3. Simulation parameters
    // -----------------------------
    double force = 4.0;
    double comVelocity = 0.0;
    double dt = 0.1;
    double threshold = 0.002;

    int numberOfSteps = 10;

    cout << "Initial COM: " << oldCOM << '\n';
    cout << "-----------------------------" << '\n';

    // -----------------------------
    // 4. Time-step loop
    // -----------------------------
    for (int step = 0; step < numberOfSteps; step++) {

        // Force -> acceleration
        double acceleration = force / totalMass;

        // Update COM velocity
        comVelocity = comVelocity + acceleration * dt;

        // Translation during this time step
        double dx = comVelocity * dt;

        // Simplified model:
        // move every ligand atom by the same amount
        x1 = x1 + dx;
        x2 = x2 + dx;
        x3 = x3 + dx;

        // Calculate the new COM
        double newCOM =
            (mass1 * x1 + mass2 * x2 + mass3 * x3)
            / totalMass;

        // COM displacement during this interval
        double displacement = fabs(newCOM - oldCOM);

        cout << "Step: " << step
             << "  COM: " << newCOM
             << "  Displacement: " << displacement
             << '\n';

        // -----------------------------
        // 5. Simplified RAMD decision
        // -----------------------------
        if (displacement < threshold) {
            cout << "Progress too small -> change direction needed."
                 << '\n';
        } else {
            cout << "Progress sufficient -> keep direction."
                 << '\n';
        }

        cout << "-----------------------------" << '\n';

        // Current COM becomes the reference for the next interval
        oldCOM = newCOM;
    }

    return 0;
}
