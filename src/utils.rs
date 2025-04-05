use image::RgbaImage;
use metal::*;
use std::path::Path;

// Simple function for loading images
pub fn load_image(path: &str) -> RgbaImage {
    let img = image::open(&Path::new(path)).unwrap_or_else(|err| {
        panic!("Failed to load image:\n{}", err);
    });
    return img.to_rgba8();
}

// Creating texture for Metal
pub fn create_texture(device: &Device, image: &RgbaImage) -> Texture {
    // Set texture width and height
    let width = image.width();
    let height = image.height();

    // Formatting texture
    let texture_desc = TextureDescriptor::new();
    texture_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    texture_desc.set_width(width as u64);
    texture_desc.set_height(height as u64);
    texture_desc.set_texture_type(MTLTextureType::D2);
    texture_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

    // Creating texture
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
    texture
}

// Saving texture to an image file
pub fn save_texture(device: &Device, texture: &Texture, width: u32, height: u32, path: &str) {
    // Create a buffer to hold the texture data
    let row_bytes = width * 4; // RGBA8 = 4 bytes per pixel
    let buffer_size = (row_bytes * height) as u64;
    
    let buffer = device.new_buffer(
        buffer_size,
        metal::MTLResourceOptions::StorageModeShared
    );
    
    // Create a command buffer for the copy
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    
    // Use a blit encoder to copy texture to buffer
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.copy_from_texture_to_buffer(
        texture, 
        0, 
        0, 
        metal::MTLOrigin{x: 0, y: 0, z: 0}, 
        metal::MTLSize::new(width as u64, height as u64, 1),
        &buffer, 
        0, 
        row_bytes as u64, 
        0,
        metal::MTLBlitOption::empty()
    );
    blit_encoder.end_encoding();
    
    // Execute the command buffer
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    // Access the buffer contents
    let buffer_ptr = buffer.contents() as *const u8;
    let data = unsafe { std::slice::from_raw_parts(buffer_ptr, buffer_size as usize) };
    
    // Create an image from the data
    let image = image::RgbaImage::from_raw(width, height, data.to_vec())
        .expect("Failed to create image from texture data");
        
    // Save the image
    image.save(path).expect("Failed to save image");
}
