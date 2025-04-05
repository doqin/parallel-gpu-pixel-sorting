use std::fs;
use metal::{Device, CommandQueue, Texture, MTLSize, MTLOrigin};

// Define TILE_SIZE to match shader
const TILE_SIZE: u32 = 256;

pub fn run_compute_shader(
    device: &Device,
    command_queue: &CommandQueue,
    input_texture: &Texture,
    width: u32,
    height: u32,
) {
    // Create a second texture for the output
    let desc = metal::TextureDescriptor::new();
    desc.set_width(input_texture.width() as u64);
    desc.set_height(input_texture.height() as u64);
    desc.set_pixel_format(input_texture.pixel_format());
    desc.set_texture_type(input_texture.texture_type());
    desc.set_usage(input_texture.usage());
    let output_texture = device.new_texture(&desc);
    
    // Read the shader from file
    let shader_path = "src/shaders/pixel_sorting.metal";
    let shader_source = fs::read_to_string(shader_path)
        .unwrap_or_else(|err| panic!("Failed to read shader file: {}", err));
    
    // Compile the shader
    let compile_options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(&shader_source, &compile_options)
        .unwrap_or_else(|err| panic!("Failed to compile shader: {}", err));
    
    // Get the function
    let function = library.get_function("sort_tile", None)
        .unwrap_or_else(|err| panic!("Failed to find shader function: {}", err));
    
    // Create the pipeline
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|err| panic!("Failed to create compute pipeline: {}", err));
    
    // Create a command buffer
    let command_buffer = command_queue.new_command_buffer();
    
    // Create an encoder
    let compute_encoder = command_buffer.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(&pipeline_state);
    
    // Set textures and parameters
    compute_encoder.set_texture(0, Some(input_texture));
    compute_encoder.set_texture(1, Some(&output_texture));
    
    // Set buffer parameters
    let width_bytes = width.to_ne_bytes();
    let height_bytes = height.to_ne_bytes();
    compute_encoder.set_bytes(0, 4, &width_bytes as *const _ as *const _);
    compute_encoder.set_bytes(1, 4, &height_bytes as *const _ as *const _);
    
    // Calculate grid size
    let tile_count = (width + TILE_SIZE - 1) / TILE_SIZE;
    
    // Set up thread execution
    let threads_per_group = MTLSize::new(TILE_SIZE as u64, 1, 1);
    let thread_groups = MTLSize::new(tile_count as u64, height as u64, 1);
    
    // Dispatch the compute kernel
    compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    compute_encoder.end_encoding();
    
    // If you need to do a merge pass, add it here with another encoder
    
    // Create a blit encoder to copy the result back to the input texture
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.copy_from_texture(
        &output_texture, 0, 0, MTLOrigin{x: 0, y: 0, z: 0},
        MTLSize::new(width as u64, height as u64, 1),
        input_texture, 0, 0, MTLOrigin{x: 0, y: 0, z: 0}
    );
    blit_encoder.end_encoding();
    
    // Execute the command buffer
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    // Save the result
    crate::utils::save_texture(device, input_texture, width, height, "output/output.png");
}
