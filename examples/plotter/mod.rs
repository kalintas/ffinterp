use eframe::egui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};

pub struct PlotData {
    pub real: Vec<[f64; 2]>,
    pub interp: Vec<[f64; 2]>,
    pub points: Vec<[f64; 2]>,
}

struct InterpolationApp {
    data: PlotData,
}

impl eframe::App for InterpolationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Interpolation Viewer");

            Plot::new("plot")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Real", PlotPoints::new(self.data.real.clone()))
                            .color(egui::Color32::RED)
                            .width(2.0),
                    );
                    plot_ui.line(
                        Line::new("Interpolated", PlotPoints::new(self.data.interp.clone()))
                            .color(egui::Color32::BLUE)
                            .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                    );
                    plot_ui.points(
                        Points::new("Input", PlotPoints::new(self.data.points.clone()))
                            .shape(MarkerShape::Circle)
                            .radius(4.0)
                            .color(egui::Color32::WHITE),
                    );
                });
        });
    }
}

pub fn show_plot(data: PlotData) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Plotter",
        options,
        Box::new(|_cc| Ok(Box::new(InterpolationApp { data }))),
    )
}
