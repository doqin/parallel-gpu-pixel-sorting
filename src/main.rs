use image::{ImageBuffer, RgbaImage};
use metal::*;
use std::fs;
use std::path::Path;

fn load_image(path: &str) -> RgbaImage {
    let img = image::open(&Path::new(path)).expect("Failed to load image");
    return img.to_rgba8();
}

fn create_texture(device: &Device, image: &RgbaImage) -> Texture {
    let width = image.width();
    let height = image.height();

    let texture_desc = TextureDescriptor::new();
    texture_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    texture_desc.set_width(width as u64);
    texture_desc.set_height(height as u64);
    texture_desc.set_texture_type(MTLTextureType::D2);
    texture_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

    let texture = device.new_texture(&texture_desc);
    texture.replace_region(
        MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        },
        0,
        image.as_raw().as_ptr() as *const _,
        (4 * width) as u64,
    );
    return texture;
}

fn run_compute_shader(
    device: &Device,
    command_queue: &CommandQueue,
    texture: &Texture,
    width: u32,
    height: u32,
) {
    let bytes_per_pixel = 4; // RGBA8
    let bytes_per_row = width * bytes_per_pixel;
    let buffer_length = (bytes_per_row * height) as usize;

    let buffer = device.new_buffer(
        buffer_length as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Create a compute pipeline using the Metal shader
    let compile_options = metal::CompileOptions::new();

    // Read shader source from file
    let shader_path = "src/shaders/pixel-sorting.metal";
    let shader_source = fs::read_to_string(shader_path)
        .unwrap_or_else(|err| {
            panic!("Failed to read shader file: {}\n{}", shader_path, err);
        });

    let library = device
        .new_library_with_source(&shader_source, &compile_options)
        .unwrap_or_else(|err| {
            panic!("Failed to create library from Metal shader source:\n{}", err);
        });

    let function = library.get_function("sort_row", None).unwrap();

    let pipeline_descriptor = ComputePipelineDescriptor::new();
    pipeline_descriptor.set_compute_function(Some(&function));

    let pipeline = device
        .new_compute_pipeline_state(&pipeline_descriptor)
        .unwrap_or_else(|err| {
            panic!("Failed to create compute pipeline state:\n{}", err);
        });

    // Create a command buffer
    let command_buffer = command_queue.new_command_buffer();

    // Create the command encoder
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    // Set buffers for texture, width, and row index
    encoder.set_texture(0, Some(texture.as_ref()));
    let width_bytes = width.to_ne_bytes();
    let height_bytes = height.to_ne_bytes();
    encoder.set_bytes(0, 4, width_bytes.as_ptr() as *const _); // Note: changed buffer index to 0
    encoder.set_bytes(1, 4, height_bytes.as_ptr() as *const _); // Note: changed buffer index to 1

    // For the tile-based approach (if implementing):
    let max_tile_size = 1024; // Must match the constant in the shader
    let num_tiles_x = (width + max_tile_size - 1) / max_tile_size; // Ceiling division

    let threads_per_group = MTLSize {
        width: max_tile_size as u64,
        height: 1,
        depth: 1,
    };

    let groups = MTLSize {
        width: num_tiles_x as u64,
        height: height as u64,
        depth: 1,
    };

    // Dispatch the compute shader
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();

    // Copy texture to buffer
    let blit_encoder = command_buffer.new_blit_command_encoder();
    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    blit_encoder.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        region.origin,
        region.size,
        &buffer,
        0, // (?)
        bytes_per_row as u64,
        bytes_per_row as u64 * height as u64,
        metal::MTLBlitOption::empty(),
    );
    blit_encoder.end_encoding();

    // Finalize
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Extract buffer into image
    let raw_data = buffer.contents();
    let pixel_slice: &[u8] =
        unsafe { std::slice::from_raw_parts(raw_data as *const u8, buffer_length) };

    let img_buf: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::from_raw(width, height, pixel_slice.to_vec())
        .expect("Failed to convert to image buffer");

    img_buf.save("output/sorted_output.png").unwrap();
}

fn main() {
    let img = load_image("input/input2.png");
    let width = img.width();
    let height = img.height();

    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();

    let texture = create_texture(&device, &img);

    run_compute_shader(&device, &command_queue, &texture, width, height);
}
