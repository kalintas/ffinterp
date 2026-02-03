use glow::HasContext;
use imgui::{Condition, Context, MouseButton};
use imgui_glow_renderer::AutoRenderer;
use imgui_sdl2_support::SdlPlatform;
use sdl2::event::Event;
use sdl2::video::GLProfile;

/// Output data from the interpolation
pub struct PlotData {
    pub real: Vec<[f64; 2]>,
    pub interp: Vec<[f64; 2]>,
    pub points: Vec<[f64; 2]>,
    /// Mean squared error between real and interpolant
    pub mse: f64,
    /// Integral of the interpolant (using trapezoidal rule)
    pub integral: f64,
    /// Hausdorff distance between interpolant and real function
    pub hausdorff: f64,
}

/// Input settings for the interpolation
#[derive(Clone)]
pub struct PlotSettings {
    /// Number of sample points
    pub n: usize,
    /// Super-sampling factor (test points per gap)
    pub test_factor: usize,
    /// Start of the x range
    pub range_start: f64,
    /// End of the x range
    pub range_end: f64,
    /// Selected function index (0-5)
    pub selected_function: usize,
    /// Whether to use individual free variables or a single scalar
    pub use_individual_d: bool,
    /// Single scalar free variable (when use_individual_d is false)
    pub d_scalar: f64,
    /// Individual free variables d_i for each interval (when use_individual_d is true)
    /// Length should be n-1
    pub d_values: Vec<f64>,
}

impl Default for PlotSettings {
    fn default() -> Self {
        Self {
            n: 150,
            test_factor: 8,
            range_start: -std::f64::consts::PI,
            range_end: std::f64::consts::PI,
            selected_function: 0,
            use_individual_d: false,
            d_scalar: 0.15,
            d_values: Vec::new(),
        }
    }
}

impl PlotSettings {
    /// Ensure d_values has the correct length for current n
    pub fn sync_d_values(&mut self) {
        let required_len = if self.n > 1 { self.n - 1 } else { 0 };
        if self.d_values.len() != required_len {
            self.d_values.resize(required_len, self.d_scalar);
        }
    }
}

/// View state for zoom and pan
struct ViewState {
    center_x: f32,
    center_y: f32,
    zoom: f32,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center_x: 0.5,
            center_y: 0.0,
            zoom: 1.0,
        }
    }
}

impl ViewState {
    fn reset(&mut self) {
        *self = Self::default();
    }
    
    fn get_visible_range(&self) -> ([f32; 2], [f32; 2]) {
        let default_x_range = 1.0;
        let default_y_range = 4.0;
        
        let visible_x_half = (default_x_range / 2.0) / self.zoom;
        let visible_y_half = (default_y_range / 2.0) / self.zoom;
        
        let x_range = [self.center_x - visible_x_half, self.center_x + visible_x_half];
        let y_range = [self.center_y - visible_y_half, self.center_y + visible_y_half];
        
        (x_range, y_range)
    }
}

/// Main function to show the interactive plotter
/// 
/// The update_fn takes PlotSettings and returns PlotData
pub fn show_plot_interactive<F>(mut update_fn: F)
where
    F: FnMut(&PlotSettings) -> PlotData + 'static,
{
    // Initialize SDL2
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();

    let gl_attr = video.gl_attr();
    gl_attr.set_context_version(3, 3);
    gl_attr.set_context_profile(GLProfile::Core);
    gl_attr.set_double_buffer(true);
    gl_attr.set_multisample_samples(4);

    let window = video
        .window("Fractal Interpolation Lab", 1280, 768)
        .position_centered()
        .opengl()
        .resizable()
        .build()
        .unwrap();

    let gl_context = window.gl_create_context().unwrap();
    window.gl_make_current(&gl_context).unwrap();
    window.subsystem().gl_set_swap_interval(1).unwrap();

    let gl = unsafe {
        glow::Context::from_loader_function(|s| video.gl_get_proc_address(s) as *const _)
    };

    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    let mut platform = SdlPlatform::init(&mut imgui);
    let mut renderer = AutoRenderer::initialize(gl, &mut imgui).unwrap();

    // State
    let mut settings = PlotSettings::default();
    settings.sync_d_values();
    let mut data = update_fn(&settings);
    let mut event_pump = sdl.event_pump().unwrap();
    let mut view = ViewState::default();
    let mut last_drag_pos: Option<[f32; 2]> = None;
    
    // UI state for int inputs (imgui needs i32)
    let mut n_input: i32 = settings.n as i32;
    let mut factor_input: i32 = settings.test_factor as i32;
    let mut range_start_input: f32 = settings.range_start as f32;
    let mut range_end_input: f32 = settings.range_end as f32;
    let mut selected_func: usize = settings.selected_function;
    
    // Function names for the combo box
    let function_names = ["Weierstrass", "Blancmange", "Multifractal", "Takagi", "Devil's Staircase", "Sine Wave", "Wen"];
    
    // Scroll position for free variables window
    let mut d_scroll_to: Option<usize> = None;

    'main: loop {
        for event in event_pump.poll_iter() {
            platform.handle_event(&mut imgui, &event);
            if let Event::Quit { .. } = event {
                break 'main;
            }
        }

        platform.prepare_frame(&mut imgui, &window, &event_pump);
        let ui = imgui.new_frame();
        
        let mut needs_update = false;

        let [display_w, display_h] = ui.io().display_size;
        let settings_w = display_w * 0.35;
        let plot_w = display_w - settings_w;

        // 1. Settings Window
        ui.window("Settings")
            .position([0.0, 0.0], Condition::Always)
            .size([settings_w, display_h], Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .build(|| {
                // Sampling parameters
                ui.text("Sampling Parameters:");
                
                ui.set_next_item_width(100.0);
                if ui.input_int("Sample Points (n)", &mut n_input).build() {
                    n_input = n_input.clamp(3, 1000);
                    settings.n = n_input as usize;
                    settings.sync_d_values();
                    needs_update = true;
                }
                
                ui.set_next_item_width(100.0);
                if ui.input_int("Test Factor", &mut factor_input).build() {
                    factor_input = factor_input.clamp(1, 10000);
                    settings.test_factor = factor_input as usize;
                    needs_update = true;
                }
                
                ui.separator();
                ui.text("X Range:");
                
                ui.set_next_item_width(100.0);
                if ui.input_float("Range Start", &mut range_start_input).build() {
                    settings.range_start = range_start_input as f64;
                    needs_update = true;
                }
                
                ui.set_next_item_width(100.0);
                if ui.input_float("Range End", &mut range_end_input).build() {
                    settings.range_end = range_end_input as f64;
                    needs_update = true;
                }
                
                ui.separator();
                
                ui.text("Function:");
                ui.set_next_item_width(150.0);
                let preview = function_names.get(selected_func).unwrap_or(&"Unknown");
                if let Some(_token) = ui.begin_combo("##func_combo", *preview) {
                    for (i, name) in function_names.iter().enumerate() {
                        let is_selected = i == selected_func;
                        if ui.selectable_config(*name).selected(is_selected).build() {
                            selected_func = i;
                            settings.selected_function = selected_func;
                            needs_update = true;
                        }
                    }
                }
                
                ui.separator();
                
                // Free variable mode toggle
                if ui.checkbox("Use Individual Free Variables", &mut settings.use_individual_d) {
                    settings.sync_d_values();
                    needs_update = true;
                }
                
                if !settings.use_individual_d {
                    // Single scalar mode
                    ui.text("Free Variable (d):");
                    
                    let mut d_f32 = settings.d_scalar as f32;
                    let slider_changed = ui.slider("##slider", -0.99, 0.99, &mut d_f32);
                    
                    ui.same_line();
                    ui.set_next_item_width(120.0);
                    let input_changed = ui.input_float("##input", &mut d_f32)
                        .step(0.001)
                        .step_fast(0.01)
                        .build();
                    
                    if slider_changed || input_changed {
                        d_f32 = d_f32.clamp(-0.99, 0.99);
                        settings.d_scalar = d_f32 as f64;
                        // Also update all individual values to match
                        for d in &mut settings.d_values {
                            *d = settings.d_scalar;
                        }
                        needs_update = true;
                    }
                } else {
                    ui.text_colored([0.5, 1.0, 0.5, 1.0], "See 'Free Variables' window");
                }
                
                ui.separator();
                ui.text("View Controls:");
                ui.text(format!("Zoom: {:.2}x", view.zoom));
                
                if ui.button("Reset View") {
                    view.reset();
                }
                ui.same_line();
                if ui.button("Zoom In") {
                    view.zoom *= 1.5;
                }
                ui.same_line();
                if ui.button("Zoom Out") {
                    view.zoom = (view.zoom / 1.5).max(0.1);
                }
                
                ui.separator();
                ui.text("Statistics:");
                ui.text(format!("MSE: {}", data.mse));
                ui.text(format!("Hausdorff: {}", data.hausdorff));
                //ui.text(format!("Dimension: {:.4}", data.dimension));
                ui.text(format!("Integral: {}", data.integral));
            });

        // 2. Free Variables Window (only shown when using individual d)
        if settings.use_individual_d {
            ui.window("Free Variables")
                .size([300.0, 400.0], Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("Individual d values (n-1 = {})", settings.d_values.len()));
                    
                    // Bulk operations
                    let mut bulk_val = settings.d_scalar as f32;
                    ui.set_next_item_width(100.0);
                    if ui.input_float("Set All To", &mut bulk_val).build() {
                        bulk_val = bulk_val.clamp(-0.99, 0.99);
                        for d in &mut settings.d_values {
                            *d = bulk_val as f64;
                        }
                        settings.d_scalar = bulk_val as f64;
                        needs_update = true;
                    }
                    
                    if ui.button("Randomize") {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
                        let mut rng = seed;
                        for d in &mut settings.d_values {
                            // Simple LCG random
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let rand_val = ((rng >> 33) as f64 / u32::MAX as f64) * 1.98 - 0.99;
                            *d = rand_val.clamp(-0.99, 0.99);
                        }
                        needs_update = true;
                    }
                    
                    ui.separator();
                    
                    // Scrollable list of d values
                    ui.child_window("d_list")
                        .size([0.0, 0.0])
                        .build(|| {
                            for i in 0..settings.d_values.len() {
                                let mut d_f32 = settings.d_values[i] as f32;
                                
                                ui.set_next_item_width(150.0);
                                let label = format!("d[{}]", i);
                                if ui.slider(&label, -0.99, 0.99, &mut d_f32) {
                                    settings.d_values[i] = d_f32 as f64;
                                    needs_update = true;
                                }
                                
                                ui.same_line();
                                ui.set_next_item_width(70.0);
                                let input_label = format!("##dinput{}", i);
                                if ui.input_float(&input_label, &mut d_f32).build() {
                                    d_f32 = d_f32.clamp(-0.99, 0.99);
                                    settings.d_values[i] = d_f32 as f64;
                                    needs_update = true;
                                }
                            }
                            
                            // Scroll to specific item if requested
                            if let Some(idx) = d_scroll_to.take() {
                                if idx < settings.d_values.len() {
                                    let item_height = ui.text_line_height_with_spacing();
                                    ui.set_scroll_y(idx as f32 * item_height);
                                }
                            }
                        });
                });
        }

        // Update data if settings changed
        if needs_update {
            data = update_fn(&settings);
        }

        // 3. Plot Window
        ui.window("Coordinate Plot")
            .position([settings_w, 0.0], Condition::Always)
            .size([plot_w, display_h], Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .build(|| {
                let draw_list = ui.get_window_draw_list();
                let canvas_pos = ui.cursor_screen_pos();
                let canvas_size = ui.content_region_avail();
                
                ui.invisible_button("canvas", canvas_size);
                let is_hovered = ui.is_item_hovered();
                let is_active = ui.is_item_active();
                
                let (x_range, y_range) = view.get_visible_range();
                
                // Mouse wheel zoom
                if is_hovered {
                    let wheel = ui.io().mouse_wheel;
                    if wheel != 0.0 {
                        let zoom_factor = if wheel > 0.0 { 1.2 } else { 1.0 / 1.2 };
                        let mouse_pos = ui.io().mouse_pos;
                        let rel_x = (mouse_pos[0] - canvas_pos[0]) / canvas_size[0];
                        let rel_y = (mouse_pos[1] - canvas_pos[1]) / canvas_size[1];
                        
                        let data_x = x_range[0] + rel_x * (x_range[1] - x_range[0]);
                        let data_y = y_range[1] - rel_y * (y_range[1] - y_range[0]);
                        
                        view.zoom = (view.zoom * zoom_factor).clamp(0.1, 100.0);
                        
                        let (new_x_range, new_y_range) = view.get_visible_range();
                        let new_data_x = new_x_range[0] + rel_x * (new_x_range[1] - new_x_range[0]);
                        let new_data_y = new_y_range[1] - rel_y * (new_y_range[1] - new_y_range[0]);
                        
                        view.center_x += data_x - new_data_x;
                        view.center_y += data_y - new_data_y;
                    }
                }
                
                // Mouse drag pan
                if is_active && ui.is_mouse_down(MouseButton::Left) {
                    let mouse_pos = ui.io().mouse_pos;
                    if let Some(last_pos) = last_drag_pos {
                        let dx_screen = mouse_pos[0] - last_pos[0];
                        let dy_screen = mouse_pos[1] - last_pos[1];
                        
                        let dx_data = -dx_screen / canvas_size[0] * (x_range[1] - x_range[0]);
                        let dy_data = dy_screen / canvas_size[1] * (y_range[1] - y_range[0]);
                        
                        view.center_x += dx_data;
                        view.center_y += dy_data;
                    }
                    last_drag_pos = Some(mouse_pos);
                } else {
                    last_drag_pos = None;
                }
                
                let (x_range, y_range) = view.get_visible_range();

                // Black Background
                draw_list.add_rect(
                    canvas_pos,
                    [canvas_pos[0] + canvas_size[0], canvas_pos[1] + canvas_size[1]],
                    [0.02, 0.02, 0.02, 1.0],
                ).filled(true).build();
                
                // Grid lines
                let grid_color = [0.2, 0.2, 0.2, 1.0];
                
                let x_step = calculate_grid_step(x_range[1] - x_range[0]);
                let x_start = (x_range[0] / x_step).floor() * x_step;
                let mut x = x_start;
                while x <= x_range[1] {
                    let screen_x = canvas_pos[0] + (x - x_range[0]) / (x_range[1] - x_range[0]) * canvas_size[0];
                    if screen_x >= canvas_pos[0] && screen_x <= canvas_pos[0] + canvas_size[0] {
                        draw_list.add_line(
                            [screen_x, canvas_pos[1]],
                            [screen_x, canvas_pos[1] + canvas_size[1]],
                            grid_color,
                        ).build();
                    }
                    x += x_step;
                }
                
                let y_step = calculate_grid_step(y_range[1] - y_range[0]);
                let y_start = (y_range[0] / y_step).floor() * y_step;
                let mut y = y_start;
                while y <= y_range[1] {
                    let screen_y = canvas_pos[1] + (1.0 - (y - y_range[0]) / (y_range[1] - y_range[0])) * canvas_size[1];
                    if screen_y >= canvas_pos[1] && screen_y <= canvas_pos[1] + canvas_size[1] {
                        draw_list.add_line(
                            [canvas_pos[0], screen_y],
                            [canvas_pos[0] + canvas_size[0], screen_y],
                            grid_color,
                        ).build();
                    }
                    y += y_step;
                }

                // Coordinate mapping
                let map = |p: [f64; 2]| -> [f32; 2] {
                    let x = canvas_pos[0] + ((p[0] as f32 - x_range[0]) / (x_range[1] - x_range[0])) * canvas_size[0];
                    let y_norm = (p[1] as f32 - y_range[0]) / (y_range[1] - y_range[0]);
                    let y = canvas_pos[1] + (1.0 - y_norm) * canvas_size[1];
                    [x, y]
                };

                if !data.real.is_empty() {
                    // Real Function (Red)
                    for win in data.real.windows(2) {
                        draw_list.add_line(map(win[0]), map(win[1]), [0.8, 0.2, 0.2, 1.0]).build();
                    }
                    // Interpolant (Cyan)
                    for win in data.interp.windows(2) {
                        draw_list.add_line(map(win[0]), map(win[1]), [0.2, 0.8, 1.0, 1.0])
                            .thickness(1.5)
                            .build();
                    }
                    // Original points (White)
                    for &p in &data.points {
                        draw_list.add_circle(map(p), 2.5, [1.0, 1.0, 1.0, 1.0])
                            .filled(true)
                            .build();
                    }
                }
                
                // Info overlay
                draw_list.add_text(
                    [canvas_pos[0] + 5.0, canvas_pos[1] + 5.0],
                    [1.0, 1.0, 1.0, 0.7],
                    format!("Zoom: {:.1}x | Scroll to zoom, drag to pan", view.zoom),
                );
            });

        // Render
        let draw_data = imgui.render();
        unsafe {
            renderer.gl_context().clear_color(0.1, 0.1, 0.1, 1.0);
            renderer.gl_context().clear(glow::COLOR_BUFFER_BIT);
        }
        renderer.render(draw_data).unwrap();
        window.gl_swap_window();
    }
}

fn calculate_grid_step(range: f32) -> f32 {
    let target_lines = 8.0;
    let raw_step = range / target_lines;
    let magnitude = 10f32.powf(raw_step.log10().floor());
    let normalized = raw_step / magnitude;
    
    let nice_step = if normalized < 1.5 {
        1.0
    } else if normalized < 3.5 {
        2.0
    } else if normalized < 7.5 {
        5.0
    } else {
        10.0
    };
    
    nice_step * magnitude
}
