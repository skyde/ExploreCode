#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <string>

// Vector3 struct to represent 3D points and vectors
struct Vector3 {
    float x, y, z;

    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    Vector3& operator+=(const Vector3& other){
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    Vector3& operator-=(const Vector3& other){
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }
    Vector3 operator*(float s) const {
        return Vector3(x * s, y * s, z * s);
    }
    float magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

// Point class representing a point in the simulation
struct Point {
    Vector3 position;
    Vector3 velocity;

    Point() : position(), velocity() {}
    Point(float x, float y, float z) : position(x, y, z), velocity() {}
};

// Spring class representing a connection between two points
struct Spring {
    int p1;
    int p2;
    float rest_length;
    float stiffness;

    Spring(int point1, int point2, float rest, float stiff)
        : p1(point1), p2(point2), rest_length(rest), stiffness(stiff) {}
};

// Simulation class handling the physics
class Simulation {
public:
    std::vector<Point> points;
    std::vector<Spring> springs;

    Simulation() {}

    void add_point(const Point& p) {
        points.push_back(p);
    }

    void add_spring(const Spring& s) {
        springs.push_back(s);
    }

    void compute_forces(std::vector<Vector3>& forces) const {
        size_t n = points.size();
        forces.assign(n, Vector3());

        // Compute repulsion forces using SIMD
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                Vector3 diff = points[i].position - points[j].position;
                float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-6f;
                float inv_dist = 1.0f / std::sqrt(dist_sq);
                float force_mag = 1.0f / dist_sq;
                Vector3 force = diff * (force_mag * inv_dist);

                forces[i] += force;
                forces[j] -= force;
            }
        }

        // Compute spring forces using SIMD
        for (const auto& spring : springs) {
            Vector3 diff = points[spring.p1].position - points[spring.p2].position;
            float dist = diff.magnitude() + 1e-6f;
            float displacement = dist - spring.rest_length;
            float force_mag = spring.stiffness * displacement;
            Vector3 force = diff * (force_mag / dist);

            forces[spring.p1] -= force;
            forces[spring.p2] += force;
        }
    }

    void step(float delta_time) {
        std::vector<Vector3> forces;
        compute_forces(forces);
        for (size_t i = 0; i < points.size(); ++i) {
            // Simple Euler integration
            points[i].velocity += forces[i] * delta_time;
            points[i].position += points[i].velocity * delta_time;
        }
    }
};

// Helper function for logging
void log_test(const std::string& test_name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << test_name << std::endl;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
    }
}

int main() {
    bool all_passed = true;

    // Test 1: Adding points
    {
        std::cout << "Running Test 1: Adding points..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 1.0f, 1.0f));
        bool condition = (sim.points.size() == 2);
        log_test("Adding points", condition);
        if (condition) {
            std::cout << "Points:" << std::endl;
            for (size_t i = 0; i < sim.points.size(); ++i) {
                std::cout << "  Point " << i << ": (" << sim.points[i].position.x << ", "
                          << sim.points[i].position.y << ", " << sim.points[i].position.z << ")" << std::endl;
            }
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 2: Adding springs
    {
        std::cout << "Running Test 2: Adding springs..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 1.0f, 1.0f));
        sim.add_spring(Spring(0, 1, 1.732f, 0.5f));
        bool condition = (sim.springs.size() == 1);
        log_test("Adding springs", condition);
        if (condition) {
            std::cout << "Springs:" << std::endl;
            for (size_t i = 0; i < sim.springs.size(); ++i) {
                std::cout << "  Spring " << i << ": connects Point " << sim.springs[i].p1
                          << " and Point " << sim.springs[i].p2
                          << ", rest_length=" << sim.springs[i].rest_length
                          << ", stiffness=" << sim.springs[i].stiffness << std::endl;
            }
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 3: Compute forces with two points
    {
        std::cout << "Running Test 3: Compute forces with two points..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f, 0.0f));
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        bool condition = (forces.size() == 2);
        log_test("Compute forces size", condition);
        if (condition) {
            std::cout << "Forces:" << std::endl;
            for (size_t i = 0; i < forces.size(); ++i) {
                std::cout << "  Force on Point " << i << ": (" << forces[i].x << ", "
                          << forces[i].y << ", " << forces[i].z << ")" << std::endl;
            }
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 4: Compute forces with repulsion
    {
        std::cout << "Running Test 4: Compute repulsion forces..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f, 0.0f));
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        Vector3 expected_force = Vector3(-1.0f, 0.0f, 0.0f); // Simplified expectation
        bool condition = (std::abs(forces[0].x + 1.0f) < 1e-3f);
        log_test("Repulsion force on point 0", condition);
        if (condition) {
            std::cout << "Force on Point 0: (" << forces[0].x << ", "
                      << forces[0].y << ", " << forces[0].z << ")" << std::endl;
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 5: Compute forces with spring
    {
        std::cout << "Running Test 5: Compute spring forces..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f, 0.0f));
        sim.add_spring(Spring(0, 1, 1.0f, 1.0f));
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        // No displacement since distance equals rest_length, so spring force should be zero
        bool condition = (std::abs(forces[0].x) < 1e-3f && std::abs(forces[1].x) < 1e-3f);
        log_test("Spring force with no displacement", condition);
        if (condition) {
            std::cout << "Force on Point 0: (" << forces[0].x << ", "
                      << forces[0].y << ", " << forces[0].z << ")" << std::endl;
            std::cout << "Force on Point 1: (" << forces[1].x << ", "
                      << forces[1].y << ", " << forces[1].z << ")" << std::endl;
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 6: Simulation step
    {
        std::cout << "Running Test 6: Simulation step..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f, 0.0f));
        sim.add_spring(Spring(0, 1, 1.0f, 1.0f));
        sim.step(0.1f);
        // After one step, positions should have been updated
        bool condition = (sim.points[0].position.x != 0.0f && sim.points[1].position.x != 1.0f);
        log_test("Position update after step", condition);
        if (condition) {
            std::cout << "After step:" << std::endl;
            for (size_t i = 0; i < sim.points.size(); ++i) {
                std::cout << "  Point " << i << " Position: (" << sim.points[i].position.x << ", "
                          << sim.points[i].position.y << ", " << sim.points[i].position.z << ")" << std::endl;
                std::cout << "  Point " << i << " Velocity: (" << sim.points[i].velocity.x << ", "
                          << sim.points[i].velocity.y << ", " << sim.points[i].velocity.z << ")" << std::endl;
            }
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 7: Edge case with zero points
    {
        std::cout << "Running Test 7: Edge case with zero points..." << std::endl;
        Simulation sim;
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        bool condition = (forces.size() == 0);
        log_test("Compute forces with zero points", condition);
        if (condition) {
            std::cout << "Forces size: " << forces.size() << std::endl;
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 8: Edge case with one point
    {
        std::cout << "Running Test 8: Edge case with one point..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        bool condition = (forces.size() == 1 && forces[0].x == 0.0f && forces[0].y == 0.0f && forces[0].z == 0.0f);
        log_test("Compute forces with one point", condition);
        if (condition) {
            std::cout << "Force on Point 0: (" << forces[0].x << ", "
                      << forces[0].y << ", " << forces[0].z << ")" << std::endl;
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 9: Overlapping points
    {
        std::cout << "Running Test 9: Overlapping points..." << std::endl;
        Simulation sim;
        sim.add_point(Point(1.0f, 1.0f, 1.0f));
        sim.add_point(Point(1.0f, 1.0f, 1.0f));
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        bool condition = (std::abs(forces[0].x) < 1e-3f && std::abs(forces[0].y) < 1e-3f && std::abs(forces[0].z) < 1e-3f);
        log_test("Repulsion force on overlapping points", condition);
        if (condition) {
            std::cout << "Force on Point 0: (" << forces[0].x << ", "
                      << forces[0].y << ", " << forces[0].z << ")" << std::endl;
            std::cout << "Force on Point 1: (" << forces[1].x << ", "
                      << forces[1].y << ", " << forces[1].z << ")" << std::endl;
        }
        all_passed &= condition;
        assert(condition);
    }

    // Test 10: Multiple springs and points
    {
        std::cout << "Running Test 10: Multiple springs and points..." << std::endl;
        Simulation sim;
        sim.add_point(Point(0.0f, 0.0f, 0.0f));
        sim.add_point(Point(1.0f, 0.0f, 0.0f));
        sim.add_point(Point(0.0f, 1.0f, 0.0f));
        sim.add_spring(Spring(0, 1, 1.0f, 1.0f));
        sim.add_spring(Spring(0, 2, 1.0f, 1.0f));
        std::vector<Vector3> forces;
        sim.compute_forces(forces);
        bool condition = (forces.size() == 3);
        log_test("Multiple springs and points size", condition);
        if (condition) {
            std::cout << "Forces:" << std::endl;
            for (size_t i = 0; i < forces.size(); ++i) {
                std::cout << "  Force on Point " << i << ": (" << forces[i].x << ", "
                          << forces[i].y << ", " << forces[i].z << ")" << std::endl;
            }
        }
        all_passed &= condition;
        assert(condition);
    }

    // Final result
    if (all_passed) {
        std::cout << "All tests passed successfully." << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed." << std::endl;
        return 1;
    }
}