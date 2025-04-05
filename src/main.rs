use metal::Device;
mod shaders;
mod utils;

fn main() {
    let img = utils::load_image("input/input3.png");
    let width = img.width();
    let height = img.height();

    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();

    let texture = utils::create_texture(&device, &img);

    shaders::run_compute_shader(&device, &command_queue, &texture, width, height);
}
