#include <metal_stdlib>
using namespace metal;

// Maximum number of pixels per tile
constant uint MAX_TILE_SIZE = 1024;

kernel void sort_row(
    texture2d<float, access::read_write> image [[texture(0)]],
    uint2 gid [[thread_position_in_grid]], 
    uint2 tid [[threadgroup_position_in_grid]],
    constant uint& width [[buffer(0)]], 
    constant uint& height [[buffer(1)]]
) {
    // Each thread group handles one row
    uint row = tid.y;

    // Skip if this row is beyond the image height
    if (row >= height) {
        return;
    }
    
    // Skip if this thread is beyond the image width
    if (gid.x >= width) {
        return;
    }
    
    // Create shared arrays for this row (with increased capacity)
    threadgroup float4 row_pixels[MAX_TILE_SIZE];
    threadgroup float brightnesses[MAX_TILE_SIZE];
    
    // Each thread loads its pixel
    uint x = gid.x;
    uint y = row;
    float4 pixel = image.read(uint2(x, y));
    float brightness = 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
    
    row_pixels[gid.x] = pixel;
    brightnesses[gid.x] = brightness;
    
    // Ensure all threads have loaded their data
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Only thread 0 sorts the pixels in this row
    if (gid.x == 0) {
        // Bubble sort implementation
        for (uint i = 0; i < min(width, MAX_TILE_SIZE); ++i) {
            for (uint j = i + 1; j < min(width, MAX_TILE_SIZE); ++j) {
                if (brightnesses[i] > brightnesses[j]) {
                    // Swap brightness values
                    float temp_b = brightnesses[i];
                    brightnesses[i] = brightnesses[j];
                    brightnesses[j] = temp_b;
                    
                    // Swap pixel values
                    float4 temp_p = row_pixels[i];
                    row_pixels[i] = row_pixels[j];
                    row_pixels[j] = temp_p;
                }
            }
        }
    }
    
    // Wait for sorting to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write back the sorted pixel
    image.write(row_pixels[gid.x], uint2(x, y));
}