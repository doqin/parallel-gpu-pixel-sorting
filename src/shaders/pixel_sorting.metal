#include <metal_stdlib>
using namespace metal;

// Maximum tile size that will fit in threadgroup memory
constant uint MAX_TILE_SIZE = 256;

kernel void sort_tile(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]], // Global thread position
    uint2 tid [[thread_position_in_threadgroup]], // Local thread position
    uint2 bid [[threadgroup_position_in_grid]], // Block/threadgroup ID
    constant uint& width [[buffer(0)]],
    constant uint& height [[buffer(1)]]
) {
    // Each threadgroup handles one row
    uint row = bid.y;
    uint tile_start = bid.x * MAX_TILE_SIZE;
    
    // Skip if this row is beyond the image bounds
    if (row >= height) return;
    
    // Create shared memory for this tile
    threadgroup float4 shared_pixels[MAX_TILE_SIZE];
    threadgroup float shared_brightness[MAX_TILE_SIZE];
    
    // Each thread loads one pixel (if in bounds)
    uint local_x = tid.x;
    uint global_x = tile_start + local_x;
    
    // Skip if this pixel is beyond the image width
    if (global_x >= width) return;
    
    // Load pixel data
    float4 pixel = input_texture.read(uint2(global_x, row));
    
    // Calculate brightness
    float brightness = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
    
    // Store in shared memory
    shared_pixels[local_x] = pixel;
    shared_brightness[local_x] = brightness;
    
    // Make sure all threads have loaded their data
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform a bitonic sort within the tile (leveraging thread parallelism)
    for (uint k = 2; k <= MAX_TILE_SIZE; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint i = tid.x;
            uint ij = i ^ j;
            
            // Only process if the xor'd index is within bounds
            if (ij > i && ij < MAX_TILE_SIZE && (tile_start + ij) < width) {
                bool ascending = ((i & k) == 0);
                if ((ascending && shared_brightness[i] > shared_brightness[ij]) ||
                    (!ascending && shared_brightness[i] < shared_brightness[ij])) {
                    // Swap
                    float temp_b = shared_brightness[i];
                    shared_brightness[i] = shared_brightness[ij];
                    shared_brightness[ij] = temp_b;
                    
                    float4 temp_p = shared_pixels[i];
                    shared_pixels[i] = shared_pixels[ij];
                    shared_pixels[ij] = temp_p;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Write sorted data back to output texture
    if (global_x < width) {
        output_texture.write(shared_pixels[local_x], uint2(global_x, row));
    }
}