#include "voxel_core.h"
#include <algorithm>
#include <cstring>

namespace voxel {

// VoxelGrid implementation
VoxelGrid::VoxelGrid(int nx, int ny, int nz, float voxel_size, const Vec3& grid_min)
    : nx_(nx), ny_(ny), nz_(nz), voxel_size_(voxel_size), grid_min_(grid_min) {
    grid_max_ = Vec3(
        grid_min_.x + nx_ * voxel_size_,
        grid_min_.y + ny_ * voxel_size_,
        grid_min_.z + nz_ * voxel_size_
    );
    data_.resize(nx_ * ny_ * nz_, 0.0f);
}

float VoxelGrid::get(int ix, int iy, int iz) const {
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) {
        return 0.0f;
    }
    return data_[index(ix, iy, iz)];
}

void VoxelGrid::set(int ix, int iy, int iz, float value) {
    if (ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_) {
        data_[index(ix, iy, iz)] = value;
    }
}

void VoxelGrid::add(int ix, int iy, int iz, float value) {
    if (ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_) {
        data_[index(ix, iy, iz)] += value;
    }
}

bool VoxelGrid::world_to_voxel(const Vec3& world_pos, int& ix, int& iy, int& iz) const {
    float fx = (world_pos.x - grid_min_.x) / voxel_size_;
    float fy = (world_pos.y - grid_min_.y) / voxel_size_;
    float fz = (world_pos.z - grid_min_.z) / voxel_size_;
    
    ix = static_cast<int>(std::floor(fx));
    iy = static_cast<int>(std::floor(fy));
    iz = static_cast<int>(std::floor(fz));
    
    return (ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_);
}

Vec3 VoxelGrid::voxel_to_world(int ix, int iy, int iz) const {
    return Vec3(
        grid_min_.x + (ix + 0.5f) * voxel_size_,
        grid_min_.y + (iy + 0.5f) * voxel_size_,
        grid_min_.z + (iz + 0.5f) * voxel_size_
    );
}

void VoxelGrid::clear() {
    std::fill(data_.begin(), data_.end(), 0.0f);
}

// RayCaster implementation
RayCaster::RayCaster(const VoxelGrid& grid)
    : grid_min_(grid.grid_min()),
      grid_max_(grid.grid_max()),
      nx_(grid.nx()),
      ny_(grid.ny()),
      nz_(grid.nz()),
      voxel_size_(grid.voxel_size()) {
}

RayAABBIntersection RayCaster::intersect_grid(const Vec3& ray_origin, const Vec3& ray_dir) const {
    RayAABBIntersection result;
    result.hit = false;
    result.t_min = 0.0f;
    result.t_max = std::numeric_limits<float>::infinity();
    
    float t_min = -std::numeric_limits<float>::infinity();
    float t_max = std::numeric_limits<float>::infinity();
    
    // Test intersection with each axis-aligned slab
    for (int i = 0; i < 3; i++) {
        float origin = (i == 0) ? ray_origin.x : ((i == 1) ? ray_origin.y : ray_origin.z);
        float dir = (i == 0) ? ray_dir.x : ((i == 1) ? ray_dir.y : ray_dir.z);
        float box_min = (i == 0) ? grid_min_.x : ((i == 1) ? grid_min_.y : grid_min_.z);
        float box_max = (i == 0) ? grid_max_.x : ((i == 1) ? grid_max_.y : grid_max_.z);
        
        if (std::abs(dir) < 1e-8f) {
            // Ray parallel to slab
            if (origin < box_min || origin > box_max) {
                return result;  // No intersection
            }
        } else {
            float t1 = (box_min - origin) / dir;
            float t2 = (box_max - origin) / dir;
            
            float t_near = std::min(t1, t2);
            float t_far = std::max(t1, t2);
            
            t_min = std::max(t_min, t_near);
            t_max = std::min(t_max, t_far);
            
            if (t_min > t_max) {
                return result;  // No intersection
            }
        }
    }
    
    result.hit = true;
    result.t_min = std::max(0.0f, t_min);
    result.t_max = t_max;
    return result;
}

void RayCaster::accumulate_ray(
    const Vec3& ray_origin,
    const Vec3& ray_dir,
    float weight,
    VoxelGrid& grid,
    float max_distance
) const {
    // Ray-grid intersection
    RayAABBIntersection isect = intersect_grid(ray_origin, ray_dir);
    if (!isect.hit) return;
    
    float t_min = isect.t_min;
    float t_max = std::min(isect.t_max, max_distance);
    
    if (t_min >= t_max) return;
    
    // DDA traversal
    Vec3 start_pos = ray_origin + ray_dir * t_min;
    
    int ix, iy, iz;
    if (!grid.world_to_voxel(start_pos, ix, iy, iz)) return;
    
    // Step directions
    int step_x = (ray_dir.x >= 0) ? 1 : -1;
    int step_y = (ray_dir.y >= 0) ? 1 : -1;
    int step_z = (ray_dir.z >= 0) ? 1 : -1;
    
    // Compute t_delta and t_max for each axis
    float t_delta_x = (std::abs(ray_dir.x) > 1e-8f) ? voxel_size_ / std::abs(ray_dir.x) : std::numeric_limits<float>::infinity();
    float t_delta_y = (std::abs(ray_dir.y) > 1e-8f) ? voxel_size_ / std::abs(ray_dir.y) : std::numeric_limits<float>::infinity();
    float t_delta_z = (std::abs(ray_dir.z) > 1e-8f) ? voxel_size_ / std::abs(ray_dir.z) : std::numeric_limits<float>::infinity();
    
    // Initial t_max values to next voxel boundary
    int next_x = ix + (step_x > 0 ? 1 : 0);
    int next_y = iy + (step_y > 0 ? 1 : 0);
    int next_z = iz + (step_z > 0 ? 1 : 0);
    
    float next_bx = grid_min_.x + next_x * voxel_size_;
    float next_by = grid_min_.y + next_y * voxel_size_;
    float next_bz = grid_min_.z + next_z * voxel_size_;
    
    float t_max_x = (std::abs(ray_dir.x) > 1e-8f) ? (next_bx - ray_origin.x) / ray_dir.x : std::numeric_limits<float>::infinity();
    float t_max_y = (std::abs(ray_dir.y) > 1e-8f) ? (next_by - ray_origin.y) / ray_dir.y : std::numeric_limits<float>::infinity();
    float t_max_z = (std::abs(ray_dir.z) > 1e-8f) ? (next_bz - ray_origin.z) / ray_dir.z : std::numeric_limits<float>::infinity();
    
    float t_current = t_min;
    
    // Traverse voxels
    while (t_current <= t_max) {
        // Accumulate weight in current voxel
        grid.add(ix, iy, iz, weight);
        
        // Step to next voxel
        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            ix += step_x;
            t_current = t_max_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_z) {
            iy += step_y;
            t_current = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_current = t_max_z;
            t_max_z += t_delta_z;
        }
        
        // Check bounds
        if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) {
            break;
        }
    }
}

void RayCaster::accumulate_rays(
    const std::vector<Vec3>& ray_origins,
    const std::vector<Vec3>& ray_dirs,
    const std::vector<float>& weights,
    VoxelGrid& grid,
    float max_distance
) const {
    size_t n = ray_origins.size();
    for (size_t i = 0; i < n; i++) {
        accumulate_ray(ray_origins[i], ray_dirs[i], weights[i], grid, max_distance);
    }
}

// Camera implementation
Camera::Camera(const CameraIntrinsics& intr, const CameraExtrinsics& extr)
    : intrinsics(intr), extrinsics(extr) {
}

void Camera::unproject_pixel(int u, int v, Vec3& ray_origin, Vec3& ray_dir) const {
    // Pixel to camera space (normalized)
    float x_cam = (u - intrinsics.cx) / intrinsics.fx;
    float y_cam = (v - intrinsics.cy) / intrinsics.fy;
    float z_cam = 1.0f;
    
    Vec3 dir_cam(x_cam, y_cam, z_cam);
    dir_cam = dir_cam.normalized();
    
    // Transform to world space
    ray_dir = extrinsics.rotation_c2w.mul(dir_cam);
    ray_dir = ray_dir.normalized();
    
    // Ray origin is camera position
    ray_origin = extrinsics.position;
}

bool Camera::project_point(const Vec3& world_pos, float& u, float& v) const {
    // Transform to camera space using world_to_camera matrix
    float world_h[4] = {world_pos.x, world_pos.y, world_pos.z, 1.0f};
    float cam[4] = {0, 0, 0, 0};
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cam[i] += extrinsics.world_to_camera[i * 4 + j] * world_h[j];
        }
    }
    
    float x = cam[0];
    float y = cam[1];
    float z = cam[2];
    
    // Check if point is in front of camera
    if (z <= 0.0f) {
        return false;
    }
    
    // Project to image plane
    u = intrinsics.fx * (x / z) + intrinsics.cx;
    v = intrinsics.fy * (y / z) + intrinsics.cy;
    
    // Check if within image bounds
    return (u >= 0 && u < intrinsics.width && v >= 0 && v < intrinsics.height);
}

// Ray crossing detection
std::vector<bool> find_crossing_rays(
    const std::vector<Vec3>& ray_origins,
    const std::vector<Vec3>& ray_dirs,
    float threshold
) {
    const size_t n = ray_origins.size();
    std::vector<bool> crossing_mask(n, false);
    
    if (n < 2) {
        return crossing_mask;
    }
    
    const float threshold_sq = threshold * threshold;
    
    // For each ray, check minimum distance to all other rays
    for (size_t i = 0; i < n; ++i) {
        const Vec3& o1 = ray_origins[i];
        const Vec3& d1 = ray_dirs[i];
        
        float a = d1.dot(d1);
        
        for (size_t j = i + 1; j < n; ++j) {
            const Vec3& o2 = ray_origins[j];
            const Vec3& d2 = ray_dirs[j];
            
            // Vector from o1 to o2
            Vec3 w = o2 - o1;
            
            // Compute dot products
            float b = d1.dot(d2);
            float c = d2.dot(d2);
            float d = d1.dot(w);
            float e = d2.dot(w);
            
            // Compute closest distance parameters
            float denom = a * c - b * b;
            
            // Skip parallel rays
            if (std::abs(denom) < 1e-8f) {
                continue;
            }
            
            float t1 = (b * e - c * d) / denom;
            float t2 = (a * e - b * d) / denom;
            
            // Compute closest points
            Vec3 p1 = o1 + d1 * t1;
            Vec3 p2 = o2 + d2 * t2;
            
            // Distance squared between closest points
            Vec3 diff = p1 - p2;
            float dist_sq = diff.dot(diff);
            
            // Mark both rays if they're within threshold
            if (dist_sq < threshold_sq) {
                crossing_mask[i] = true;
                crossing_mask[j] = true;
            }
        }
    }
    
    return crossing_mask;
}

}  // namespace voxel
