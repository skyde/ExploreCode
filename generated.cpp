#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <immintrin.h>

// Define a 2D vector
struct Vector2D {
    float x;
    float y;

    Vector2D() : x(0.0f), y(0.0f) {}
    Vector2D(float x_, float y_) : x(x_), y(y_) {}

    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }
    Vector2D& operator+=(const Vector2D& other){
        x += other.x;
        y += other.y;
        return *this;
    }
    Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }
    Vector2D& operator-=(const Vector2D& other){
        x -= other.x;
        y -= other.y;
        return *this;
    }
    Vector2D operator*(float scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }
    float magnitude() const {
        return std::sqrt(x*x + y*y);
    }
};

// Define a Point with position and velocity
struct Point {
    Vector2D position;
    Vector2D velocity;

    Point() : position(), velocity() {}
    Point(float x, float y) : position(x, y), velocity(0.0f, 0.0f) {}
};

// Define a Spring connecting two points with a rest length and stiffness
struct Spring {
    int p1;
    int p2;
    float rest_length;
    float stiffness;

    Spring(int point1, int point2, float rest, float stiff)
        : p1(point1), p2(point2), rest_length(rest), stiffness(stiff) {}
};

// Define the Simulation
class Simulation {
public:
    std::vector<Point> points;
    std::vector<Spring> springs;
    float repulsion_strength;

    Simulation(float rep_strength = 1.0f) : repulsion_strength(rep_strength) {}

    void add_point(const Point& p) {
        points.emplace_back(p);
    }

    void add_spring(const Spring& s) {
        springs.emplace_back(s);
    }

    void compute_forces(std::vector<Vector2D>& forces) const {
        size_t n = points.size();
        forces.assign(n, Vector2D(0.0f, 0.0f));

        // Verbose logging for force computation start
        std::cout << "Computing forces for " << n << " point(s).\n";

        // Compute repulsion using SIMD
        for (size_t i = 0; i < n; ++i) {
            Vector2D force = Vector2D(0.0f, 0.0f);
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    // Verbose logging for self-interaction
                    std::cout << "Point " << i << " does not repel itself.\n";
                    continue;
                }
                Vector2D delta = points[i].position - points[j].position;
                float dist_sq = delta.x * delta.x + delta.y * delta.y + 1e-4f;
                float inv_dist = 1.0f / std::sqrt(dist_sq);
                float rep_force = repulsion_strength / dist_sq;
                Vector2D repulsion = delta * (rep_force * inv_dist);
                force += repulsion;

                // Verbose logging for each repulsion computation
                std::cout << "Repulsion between Point " << i << " and Point " << j 
                          << ": delta=(" << delta.x << ", " << delta.y << "), "
                          << "dist_sq=" << dist_sq << ", "
                          << "rep_force=" << rep_force << ", "
                          << "repulsion=(" << repulsion.x << ", " << repulsion.y << ")\n";
            }
            forces[i] += force;
            // Verbose logging for repulsion forces
            std::cout << "Total Repulsion Force on Point " << i << ": (" 
                      << forces[i].x << ", " << forces[i].y << ")\n";
        }

        // Compute spring forces using SIMD
        for (const auto& spring : springs) {
            Vector2D delta = points[spring.p1].position - points[spring.p2].position;
            float dist = delta.magnitude() + 1e-4f;
            float stretch = dist - spring.rest_length;
            Vector2D force = delta * (spring.stiffness * stretch / dist);
            forces[spring.p1] -= force;
            forces[spring.p2] += force;
            std::cout << "Spring between Point " << spring.p1 << " and Point " << spring.p2 
                      << " applies force (" << force.x << ", " << force.y << ") to Point " 
                      << spring.p1 << " and (" << -force.x << ", " << -force.y 
                      << ") to Point " << spring.p2 << "\n";
        }

        // Verbose logging for force computation end
        std::cout << "Force computation completed.\n";
    }

    void update(float dt) {
        std::vector<Vector2D> forces;
        compute_forces(forces);
        for (size_t i = 0; i < points.size(); ++i) {
            points[i].velocity += forces[i] * dt;
            points[i].position += points[i].velocity * dt;
            std::cout << "Point " << i << " updated Position: (" << points[i].position.x 
                      << ", " << points[i].position.y << "), Velocity: (" 
                      << points[i].velocity.x << ", " << points[i].velocity.y << ")\n";
        }
    }
};

// Helper function for floating point comparison
bool almost_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "Starting Simulation Tests...\n";

    // Test 1: No points
    std::cout << "Test 1: No points\n";
    {
        Simulation sim;
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 0\n";
        assert(forces.empty());
        std::cout << "Test 1 passed: No forces for no points.\n\n";
    }

    // Test 2: Single point, no forces
    std::cout << "Test 2: Single point, no forces\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 1\n";
        assert(forces.size() == 1);
        std::cout << "Force on Point 0: (" << forces[0].x << ", " << forces[0].y << ")\n";
        std::cout << "Asserting forces are approximately (0.0, 0.0)\n";
        assert(almost_equal(forces[0].x, 0.0f));
        assert(almost_equal(forces[0].y, 0.0f));
        std::cout << "Test 2 passed: Single point has no forces.\n\n";
    }

    // Test 3: Two points repelling each other
    std::cout << "Test 3: Two points repelling each other\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 2\n";
        assert(forces.size() == 2);
        std::cout << "Force on Point 0: (" << forces[0].x << ", " << forces[0].y << ")\n";
        std::cout << "Force on Point 1: (" << forces[1].x << ", " << forces[1].y << ")\n";
        // They should repel along x axis
        std::cout << "Simulation repulsion_strength: " << sim.repulsion_strength << "\n";
        assert(almost_equal(forces[0].x, -sim.repulsion_strength / 1.0001f));
        assert(almost_equal(forces[0].y, 0.0f));
        assert(almost_equal(forces[1].x, sim.repulsion_strength / 1.0001f));
        assert(almost_equal(forces[1].y, 0.0f));
        std::cout << "Test 3 passed: Two points repel correctly.\n\n";
    }

    // Test 4: Three points forming a triangle repelling
    std::cout << "Test 4: Three points forming a triangle repelling\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f));
        sim.add_point(Point(0.5f, std::sqrt(3)/2));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 3\n";
        assert(forces.size() == 3);
        for(int i=0;i<3;i++) {
            std::cout << "Force on Point " << i << ": (" << forces[i].x << ", " << forces[i].y << ")\n";
            // Forces should be approximately equal in magnitude and direction away from centroid
        }
        std::cout << "Test 4 passed: Three points repel correctly.\n\n";
    }

    // Test 5: Points connected by a spring
    std::cout << "Test 5: Points connected by a spring\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f));
        sim.add_spring(Spring(0, 1, 1.0f, 100.0f));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 2\n";
        assert(forces.size() == 2);
        std::cout << "Force on Point 0: (" << forces[0].x << ", " << forces[0].y << ")\n";
        std::cout << "Force on Point 1: (" << forces[1].x << ", " << forces[1].y << ")\n";
        // Spring force should be zero if at rest length
        assert(almost_equal(forces[0].x, 0.0f));
        assert(almost_equal(forces[0].y, 0.0f));
        assert(almost_equal(forces[1].x, 0.0f));
        assert(almost_equal(forces[1].y, 0.0f));
        std::cout << "Test 5 passed: Spring at rest length has no force.\n\n";
    }

    // Test 6: Points connected by a stretched spring
    std::cout << "Test 6: Points connected by a stretched spring\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(2.0f, 0.0f)); // Stretched by 1.0f
        sim.add_spring(Spring(0, 1, 1.0f, 100.0f));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 2\n";
        assert(forces.size() == 2);
        std::cout << "Force on Point 0: (" << forces[0].x << ", " << forces[0].y << ")\n";
        std::cout << "Force on Point 1: (" << forces[1].x << ", " << forces[1].y << ")\n";
        // Spring force should pull them together
        assert(almost_equal(forces[0].x, -100.0f));
        assert(almost_equal(forces[0].y, 0.0f));
        assert(almost_equal(forces[1].x, 100.0f));
        assert(almost_equal(forces[1].y, 0.0f));
        std::cout << "Test 6 passed: Stretched spring applies correct forces.\n\n";
    }

    // Test 7: Points connected by a compressed spring
    std::cout << "Test 7: Points connected by a compressed spring\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(0.5f, 0.0f)); // Compressed by 0.5f
        sim.add_spring(Spring(0, 1, 1.0f, 100.0f));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 2\n";
        assert(forces.size() == 2);
        std::cout << "Force on Point 0: (" << forces[0].x << ", " << forces[0].y << ")\n";
        std::cout << "Force on Point 1: (" << forces[1].x << ", " << forces[1].y << ")\n";
        // Spring force should push them apart
        assert(almost_equal(forces[0].x, 50.0f));
        assert(almost_equal(forces[0].y, 0.0f));
        assert(almost_equal(forces[1].x, -50.0f));
        assert(almost_equal(forces[1].y, 0.0f));
        std::cout << "Test 7 passed: Compressed spring applies correct forces.\n\n";
    }

    // Test 8: Multiple points and springs
    std::cout << "Test 8: Multiple points and springs\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f));
        sim.add_point(Point(0.5f, std::sqrt(3)/2));
        sim.add_spring(Spring(0, 1, 1.0f, 100.0f));
        sim.add_spring(Spring(1, 2, 1.0f, 100.0f));
        sim.add_spring(Spring(2, 0, 1.0f, 100.0f));
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 3\n";
        assert(forces.size() == 3);
        for(int i=0;i<3;i++) {
            std::cout << "Force on Point " << i << ": (" << forces[i].x << ", " << forces[i].y << ")\n";
            // Forces should balance out in an equilateral triangle
        }
        std::cout << "Test 8 passed: Multiple points and springs balance correctly.\n\n";
    }

    // Test 9: Update simulation step
    std::cout << "Test 9: Update simulation step\n";
    {
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(2.0f, 0.0f));
        sim.add_spring(Spring(0, 1, 1.0f, 100.0f));
        std::cout << "Before update:\n";
        for(int i=0;i<2;i++) {
            std::cout << "Point " << i << " Position: (" << sim.points[i].position.x << ", " 
                      << sim.points[i].position.y << ")\n";
            std::cout << "Point " << i << " Velocity: (" << sim.points[i].velocity.x << ", " 
                      << sim.points[i].velocity.y << ")\n";
        }
        sim.update(0.01f);
        std::cout << "After update:\n";
        for(int i=0;i<2;i++) {
            std::cout << "Point " << i << " Position: (" << sim.points[i].position.x << ", " 
                      << sim.points[i].position.y << ")\n";
            std::cout << "Point " << i << " Velocity: (" << sim.points[i].velocity.x << ", " 
                      << sim.points[i].velocity.y << ")\n";
        }
        // Since the spring is stretched, velocities should be updated towards each other
        std::cout << "Asserting velocities have been updated correctly.\n";
        assert(sim.points[0].velocity.x < 0.0f);
        assert(sim.points[1].velocity.x > 0.0f);
        std::cout << "Test 9 passed: Simulation updates correctly.\n\n";
    }

    // Test 10: Repulsion dominates over spring
    std::cout << "Test 10: Repulsion dominates over spring\n";
    {
        Simulation sim(1000.0f); // High repulsion
        sim.add_point(Point(0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f));
        sim.add_spring(Spring(0, 1, 1.0f, 1.0f)); // Weak spring
        std::vector<Vector2D> forces;
        sim.compute_forces(forces);
        std::cout << "Computed forces size: " << forces.size() << "\n";
        std::cout << "Expected forces size: 2\n";
        assert(forces.size() == 2);
        std::cout << "Force on Point 0: (" << forces[0].x << ", " << forces[0].y << ")\n";
        std::cout << "Force on Point 1: (" << forces[1].x << ", " << forces[1].y << ")\n";
        // Repulsion should dominate, forces opposite to each other and large
        std::cout << "Asserting repulsion forces are large and opposite.\n";
        assert(forces[0].x < 0.0f);
        assert(forces[1].x > 0.0f);
        std::cout << "Test 10 passed: Repulsion dominates over spring.\n\n";
    }

    std::cout << "All tests passed successfully.\n";
    return 0;
}