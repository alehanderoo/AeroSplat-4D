#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <limits>

namespace voxel {

// 3D vector/point type
struct Vec3 {
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const {
        float len = length();
        return (len > 1e-8f) ? (*this * (1.0f / len)) : Vec3(0, 0, 0);
    }
};

// 3x3 matrix for rotations
struct Mat3 {
    float m[9];
    
    Mat3() { for (int i = 0; i < 9; i++) m[i] = 0.0f; }
    
    Vec3 mul(const Vec3& v) const {
        return Vec3(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z
        );
    }
};

// Camera intrinsics
struct CameraIntrinsics {
    float fx, fy, cx, cy;
    int width, height;
};

// Camera extrinsics (world to camera and camera to world)
struct CameraExtrinsics {
    float world_to_camera[16];  // 4x4 matrix
    float camera_to_world[16];  // 4x4 matrix
    Vec3 position;
    Mat3 rotation_c2w;  // camera to world rotation
};

// Voxel grid representation
class VoxelGrid {
public:
    VoxelGrid(int nx, int ny, int nz, float voxel_size, const Vec3& grid_min);
    
    // Get grid dimensions
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    float voxel_size() const { return voxel_size_; }
    Vec3 grid_min() const { return grid_min_; }
    Vec3 grid_max() const { return grid_max_; }
    
    // Access voxel value
    float get(int ix, int iy, int iz) const;
    void set(int ix, int iy, int iz, float value);
    void add(int ix, int iy, int iz, float value);
    
    // Convert world coordinates to voxel indices
    bool world_to_voxel(const Vec3& world_pos, int& ix, int& iy, int& iz) const;
    
    // Convert voxel indices to world coordinates (center of voxel)
    Vec3 voxel_to_world(int ix, int iy, int iz) const;
    
    // Clear grid
    void clear();
    
    // Get raw data pointer (for numpy interface)
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    
private:
    int nx_, ny_, nz_;
    float voxel_size_;
    Vec3 grid_min_;
    Vec3 grid_max_;
    std::vector<float> data_;
    
    inline int index(int ix, int iy, int iz) const {
        return ix * ny_ * nz_ + iy * nz_ + iz;
    }
};

// Ray-AABB intersection result
struct RayAABBIntersection {
    bool hit;
    float t_min;
    float t_max;
};

// Ray casting utilities
class RayCaster {
public:
    RayCaster(const VoxelGrid& grid);
    
    // Ray-AABB intersection test
    RayAABBIntersection intersect_grid(const Vec3& ray_origin, const Vec3& ray_dir) const;
    
    // DDA voxel traversal: accumulate weight along ray
    void accumulate_ray(
        const Vec3& ray_origin,
        const Vec3& ray_dir,
        float weight,
        VoxelGrid& grid,
        float max_distance = 100.0f
    ) const;
    
    // Batch processing: accumulate multiple rays
    void accumulate_rays(
        const std::vector<Vec3>& ray_origins,
        const std::vector<Vec3>& ray_dirs,
        const std::vector<float>& weights,
        VoxelGrid& grid,
        float max_distance = 100.0f
    ) const;
    
private:
    Vec3 grid_min_;
    Vec3 grid_max_;
    int nx_, ny_, nz_;
    float voxel_size_;
};

// Camera utilities
class Camera {
public:
    Camera(const CameraIntrinsics& intrinsics, const CameraExtrinsics& extrinsics);
    
    // Unproject pixel to world ray
    void unproject_pixel(int u, int v, Vec3& ray_origin, Vec3& ray_dir) const;
    
    // Project world point to pixel
    bool project_point(const Vec3& world_pos, float& u, float& v) const;
    
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
};

// Ray crossing detection
std::vector<bool> find_crossing_rays(
    const std::vector<Vec3>& ray_origins,
    const std::vector<Vec3>& ray_dirs,
    float threshold
);

}  // namespace voxel
