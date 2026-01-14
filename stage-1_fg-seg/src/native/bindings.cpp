#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "voxel_core.h"

namespace py = pybind11;

// Wrapper to create VoxelGrid from numpy shape
voxel::VoxelGrid* create_voxel_grid(
    py::array_t<int> shape,
    float voxel_size,
    py::array_t<float> grid_min
) {
    auto shape_buf = shape.request();
    auto min_buf = grid_min.request();
    
    if (shape_buf.size != 3 || min_buf.size != 3) {
        throw std::runtime_error("shape and grid_min must have 3 elements");
    }
    
    int* shape_ptr = static_cast<int*>(shape_buf.ptr);
    float* min_ptr = static_cast<float*>(min_buf.ptr);
    
    voxel::Vec3 min_vec(min_ptr[0], min_ptr[1], min_ptr[2]);
    
    return new voxel::VoxelGrid(shape_ptr[0], shape_ptr[1], shape_ptr[2], voxel_size, min_vec);
}

// Export VoxelGrid data as numpy array (zero-copy view)
py::array_t<float> get_voxel_grid_data(voxel::VoxelGrid* grid) {
    std::vector<ssize_t> shape = {
        static_cast<ssize_t>(grid->nx()),
        static_cast<ssize_t>(grid->ny()),
        static_cast<ssize_t>(grid->nz())
    };
    
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(grid->ny() * grid->nz() * sizeof(float)),
        static_cast<ssize_t>(grid->nz() * sizeof(float)),
        static_cast<ssize_t>(sizeof(float))
    };
    
    return py::array_t<float>(shape, strides, grid->data(), py::cast(grid));
}

// Set VoxelGrid data from numpy array
void set_voxel_grid_data(voxel::VoxelGrid* grid, py::array_t<float> data) {
    auto buf = data.request();
    
    if (buf.ndim != 3) {
        throw std::runtime_error("Input array must be 3D");
    }
    
    if (buf.shape[0] != grid->nx() || buf.shape[1] != grid->ny() || buf.shape[2] != grid->nz()) {
        throw std::runtime_error("Array shape mismatch");
    }
    
    float* src = static_cast<float*>(buf.ptr);
    float* dst = grid->data();
    std::memcpy(dst, src, grid->size() * sizeof(float));
}

// Cast multiple rays into grid
void cast_rays_batch(
    voxel::VoxelGrid* grid,
    py::array_t<float> ray_origins,
    py::array_t<float> ray_dirs,
    py::array_t<float> weights,
    float max_distance
) {
    auto origins_buf = ray_origins.request();
    auto dirs_buf = ray_dirs.request();
    auto weights_buf = weights.request();
    
    if (origins_buf.ndim != 2 || origins_buf.shape[1] != 3) {
        throw std::runtime_error("ray_origins must be (N, 3)");
    }
    if (dirs_buf.ndim != 2 || dirs_buf.shape[1] != 3) {
        throw std::runtime_error("ray_dirs must be (N, 3)");
    }
    if (weights_buf.ndim != 1) {
        throw std::runtime_error("weights must be (N,)");
    }
    
    size_t n = origins_buf.shape[0];
    if (dirs_buf.shape[0] != n || weights_buf.shape[0] != n) {
        throw std::runtime_error("Array size mismatch");
    }
    
    float* origins_ptr = static_cast<float*>(origins_buf.ptr);
    float* dirs_ptr = static_cast<float*>(dirs_buf.ptr);
    float* weights_ptr = static_cast<float*>(weights_buf.ptr);
    
    // Convert to C++ vectors
    std::vector<voxel::Vec3> origins_vec, dirs_vec;
    std::vector<float> weights_vec;
    
    origins_vec.reserve(n);
    dirs_vec.reserve(n);
    weights_vec.reserve(n);
    
    for (size_t i = 0; i < n; i++) {
        origins_vec.emplace_back(origins_ptr[i*3], origins_ptr[i*3+1], origins_ptr[i*3+2]);
        dirs_vec.emplace_back(dirs_ptr[i*3], dirs_ptr[i*3+1], dirs_ptr[i*3+2]);
        weights_vec.push_back(weights_ptr[i]);
    }
    
    // Cast rays
    voxel::RayCaster caster(*grid);
    caster.accumulate_rays(origins_vec, dirs_vec, weights_vec, *grid, max_distance);
}

// Unproject pixels to rays
std::tuple<py::array_t<float>, py::array_t<float>> unproject_pixels(
    py::array_t<int> pixel_coords,
    py::array_t<float> camera_to_world_rotation,
    py::array_t<float> camera_position,
    float fx, float fy, float cx, float cy
) {
    auto coords_buf = pixel_coords.request();
    auto rot_buf = camera_to_world_rotation.request();
    auto pos_buf = camera_position.request();
    
    if (coords_buf.ndim != 2 || coords_buf.shape[1] != 2) {
        throw std::runtime_error("pixel_coords must be (N, 2)");
    }
    if (rot_buf.size != 9) {
        throw std::runtime_error("rotation must be 3x3");
    }
    if (pos_buf.size != 3) {
        throw std::runtime_error("position must be (3,)");
    }
    
    size_t n = coords_buf.shape[0];
    int* coords_ptr = static_cast<int*>(coords_buf.ptr);
    float* rot_ptr = static_cast<float*>(rot_buf.ptr);
    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    
    // Setup camera
    voxel::Mat3 rot_c2w;
    std::memcpy(rot_c2w.m, rot_ptr, 9 * sizeof(float));
    
    voxel::Vec3 cam_pos(pos_ptr[0], pos_ptr[1], pos_ptr[2]);
    
    voxel::CameraIntrinsics intr;
    intr.fx = fx;
    intr.fy = fy;
    intr.cx = cx;
    intr.cy = cy;
    
    voxel::CameraExtrinsics extr;
    extr.position = cam_pos;
    extr.rotation_c2w = rot_c2w;
    
    voxel::Camera camera(intr, extr);
    
    // Allocate output arrays
    std::vector<ssize_t> shape = {static_cast<ssize_t>(n), 3};
    py::array_t<float> origins(shape);
    py::array_t<float> dirs(shape);
    
    auto origins_buf = origins.request();
    auto dirs_buf = dirs.request();
    
    float* origins_ptr = static_cast<float*>(origins_buf.ptr);
    float* dirs_ptr = static_cast<float*>(dirs_buf.ptr);
    
    // Unproject each pixel
    for (size_t i = 0; i < n; i++) {
        int u = coords_ptr[i * 2];
        int v = coords_ptr[i * 2 + 1];
        
        voxel::Vec3 ray_origin, ray_dir;
        camera.unproject_pixel(u, v, ray_origin, ray_dir);
        
        origins_ptr[i * 3] = ray_origin.x;
        origins_ptr[i * 3 + 1] = ray_origin.y;
        origins_ptr[i * 3 + 2] = ray_origin.z;
        
        dirs_ptr[i * 3] = ray_dir.x;
        dirs_ptr[i * 3 + 1] = ray_dir.y;
        dirs_ptr[i * 3 + 2] = ray_dir.z;
    }
    
    return std::make_tuple(origins, dirs);
}

// Visibility counting for voxelizer
py::array_t<int> count_voxel_visibility(
    py::array_t<float> voxel_points,
    py::list camera_list
) {
    auto points_buf = voxel_points.request();
    
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("voxel_points must be (N, 3)");
    }
    
    size_t n = points_buf.shape[0];
    float* points_ptr = static_cast<float*>(points_buf.ptr);
    
    // Allocate output
    std::vector<ssize_t> counts_shape = {static_cast<ssize_t>(n)};
    py::array_t<int> counts(counts_shape);
    auto counts_buf = counts.request();
    int* counts_ptr = static_cast<int*>(counts_buf.ptr);
    std::memset(counts_ptr, 0, n * sizeof(int));
    
    // Process each camera
    for (size_t cam_idx = 0; cam_idx < camera_list.size(); cam_idx++) {
        py::dict cam = camera_list[cam_idx].cast<py::dict>();
        
        // Extract camera parameters
        auto w2c_array = cam["world_to_camera"].cast<py::array_t<float>>();
        auto w2c_buf = w2c_array.request();
        float* w2c_ptr = static_cast<float*>(w2c_buf.ptr);
        
        float fx = cam["fx"].cast<float>();
        float fy = cam["fy"].cast<float>();
        float cx = cam["cx"].cast<float>();
        float cy = cam["cy"].cast<float>();
        float width = cam["width"].cast<float>();
        float height = cam["height"].cast<float>();
        float near = cam["near"].cast<float>();
        float far = cam["far"].cast<float>();
        
        // Check each point
        for (size_t i = 0; i < n; i++) {
            float world_h[4] = {
                points_ptr[i * 3],
                points_ptr[i * 3 + 1],
                points_ptr[i * 3 + 2],
                1.0f
            };
            
            // Transform to camera space
            float cam[4] = {0, 0, 0, 0};
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    cam[row] += w2c_ptr[row * 4 + col] * world_h[col];
                }
            }
            
            float x = cam[0];
            float y = cam[1];
            float z = cam[2];
            
            // Check depth bounds
            if (z <= near || z >= far) continue;
            
            // Project to image
            float u = fx * (x / z) + cx;
            float v = fy * (y / z) + cy;
            
            // Check image bounds
            if (u >= 0.0f && u < width && v >= 0.0f && v < height) {
                counts_ptr[i]++;
            }
        }
    }
    
    return counts;
}

// Find maximum in voxel grid and return voxel index
std::tuple<int, int, int, float> find_max_voxel(voxel::VoxelGrid* grid) {
    float max_val = -std::numeric_limits<float>::infinity();
    int max_ix = 0, max_iy = 0, max_iz = 0;
    
    for (int ix = 0; ix < grid->nx(); ix++) {
        for (int iy = 0; iy < grid->ny(); iy++) {
            for (int iz = 0; iz < grid->nz(); iz++) {
                float val = grid->get(ix, iy, iz);
                if (val > max_val) {
                    max_val = val;
                    max_ix = ix;
                    max_iy = iy;
                    max_iz = iz;
                }
            }
        }
    }
    
    return std::make_tuple(max_ix, max_iy, max_iz, max_val);
}

// Convert voxel indices to world coordinates
py::array_t<float> voxel_indices_to_world(
    voxel::VoxelGrid* grid,
    py::array_t<int> indices
) {
    auto idx_buf = indices.request();
    
    if (idx_buf.ndim != 2 || idx_buf.shape[1] != 3) {
        throw std::runtime_error("indices must be (N, 3)");
    }
    
    size_t n = idx_buf.shape[0];
    int* idx_ptr = static_cast<int*>(idx_buf.ptr);
    
    std::vector<ssize_t> world_shape = {static_cast<ssize_t>(n), 3};
    py::array_t<float> world_coords(world_shape);
    auto world_buf = world_coords.request();
    float* world_ptr = static_cast<float*>(world_buf.ptr);
    
    for (size_t i = 0; i < n; i++) {
        int ix = idx_ptr[i * 3];
        int iy = idx_ptr[i * 3 + 1];
        int iz = idx_ptr[i * 3 + 2];
        
        voxel::Vec3 world = grid->voxel_to_world(ix, iy, iz);
        
        world_ptr[i * 3] = world.x;
        world_ptr[i * 3 + 1] = world.y;
        world_ptr[i * 3 + 2] = world.z;
    }
    
    return world_coords;
}

PYBIND11_MODULE(voxel_ops, m) {
    m.doc() = "C++ voxel operations for pixel2voxel";
    
    // VoxelGrid class
    py::class_<voxel::VoxelGrid>(m, "VoxelGrid")
        .def(py::init<int, int, int, float, voxel::Vec3>())
        .def("nx", &voxel::VoxelGrid::nx)
        .def("ny", &voxel::VoxelGrid::ny)
        .def("nz", &voxel::VoxelGrid::nz)
        .def("voxel_size", &voxel::VoxelGrid::voxel_size)
        .def("get", &voxel::VoxelGrid::get)
        .def("set", &voxel::VoxelGrid::set)
        .def("add", &voxel::VoxelGrid::add)
        .def("clear", &voxel::VoxelGrid::clear)
        .def("data", &get_voxel_grid_data);
    
    // Vec3 class
    py::class_<voxel::Vec3>(m, "Vec3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &voxel::Vec3::x)
        .def_readwrite("y", &voxel::Vec3::y)
        .def_readwrite("z", &voxel::Vec3::z);
    
    // Standalone functions
    m.def("create_voxel_grid", &create_voxel_grid,
          "Create voxel grid from shape and parameters",
          py::arg("shape"), py::arg("voxel_size"), py::arg("grid_min"));
    
    m.def("set_voxel_grid_data", &set_voxel_grid_data,
          "Set voxel grid data from numpy array",
          py::arg("grid"), py::arg("data"));
    
    m.def("cast_rays_batch", &cast_rays_batch,
          "Cast multiple rays into voxel grid",
          py::arg("grid"), py::arg("ray_origins"), py::arg("ray_dirs"),
          py::arg("weights"), py::arg("max_distance") = 100.0f);
    
    m.def("unproject_pixels", &unproject_pixels,
          "Unproject pixels to world rays",
          py::arg("pixel_coords"), py::arg("camera_to_world_rotation"),
          py::arg("camera_position"), py::arg("fx"), py::arg("fy"),
          py::arg("cx"), py::arg("cy"));
    
    m.def("count_voxel_visibility", &count_voxel_visibility,
          "Count how many cameras see each voxel point",
          py::arg("voxel_points"), py::arg("camera_list"));
    
    m.def("find_max_voxel", &find_max_voxel,
          "Find voxel with maximum value",
          py::arg("grid"));
    
    m.def("voxel_indices_to_world", &voxel_indices_to_world,
          "Convert voxel indices to world coordinates",
          py::arg("grid"), py::arg("indices"));
    
    m.def("find_crossing_rays", [](
        py::array_t<float> ray_origins,
        py::array_t<float> ray_dirs,
        float threshold
    ) -> py::array_t<bool> {
        auto origins_buf = ray_origins.request();
        auto dirs_buf = ray_dirs.request();
        
        if (origins_buf.ndim != 2 || origins_buf.shape[1] != 3) {
            throw std::runtime_error("ray_origins must be (N, 3)");
        }
        if (dirs_buf.ndim != 2 || dirs_buf.shape[1] != 3) {
            throw std::runtime_error("ray_dirs must be (N, 3)");
        }
        
        size_t n = origins_buf.shape[0];
        if (dirs_buf.shape[0] != n) {
            throw std::runtime_error("Array size mismatch");
        }
        
        float* origins_ptr = static_cast<float*>(origins_buf.ptr);
        float* dirs_ptr = static_cast<float*>(dirs_buf.ptr);
        
        // Convert to C++ vectors
        std::vector<voxel::Vec3> origins_vec, dirs_vec;
        origins_vec.reserve(n);
        dirs_vec.reserve(n);
        
        for (size_t i = 0; i < n; i++) {
            origins_vec.emplace_back(origins_ptr[i*3], origins_ptr[i*3+1], origins_ptr[i*3+2]);
            dirs_vec.emplace_back(dirs_ptr[i*3], dirs_ptr[i*3+1], dirs_ptr[i*3+2]);
        }
        
        // Call C++ function
        std::vector<bool> result = voxel::find_crossing_rays(origins_vec, dirs_vec, threshold);
        
        // Convert to numpy
        py::array_t<bool> mask(n);
        auto mask_buf = mask.request();
        bool* mask_ptr = static_cast<bool*>(mask_buf.ptr);
        
        for (size_t i = 0; i < n; i++) {
            mask_ptr[i] = result[i];
        }
        
        return mask;
    },
    "Find rays that cross in 3D space",
    py::arg("ray_origins"), py::arg("ray_dirs"), py::arg("threshold") = 0.5f);
}
