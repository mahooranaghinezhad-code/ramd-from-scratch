#include <iostream>
using namespace std;

int main() {

    double mass = 2.0;
    double force = 4.0;

    double position = 0.0;
    double velocity = 0.0;

    double dt = 0.1;

    for (int step = 0; step < 100; step++) {

        double acceleration = force / mass;

        velocity = velocity + acceleration * dt;
        position = position + velocity * dt;

        cout << "Step: " << step
             << "  Velocity: " << velocity
             << "  Position: " << position
             << '\n';
    }

    return 0;
}
