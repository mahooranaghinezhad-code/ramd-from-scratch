#include <iostream>
using namespace std;

int main() {

    double mass = 2.0;
    double force = 4.0;

    double position = 0.0;
    double velocity = 0.0;

    double dt = 0.1;

    double acceleration = force / mass;

    velocity = velocity + acceleration * dt;
    position = position + velocity * dt;

    cout << "Acceleration: " << acceleration << '\n';
    cout << "Velocity: " << velocity << '\n';
    cout << "Position: " << position << '\n';

    return 0;
}
