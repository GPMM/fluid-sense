extern crate core;

use clap::Parser;
use crate::app::App;
use crate::cfd::sph::simulation::{SimulationParticle, SPH};
use crate::gfx::buffer::VertexBuffer;
use crate::gfx::camera::controller::FirstPersonController;
use crate::gfx::camera::projection::Perspective;
use crate::gfx::camera::Camera;
use crate::gfx::light::Light;
use crate::gfx::pipeline::Pipeline;
use crate::gfx::renderer::Renderer;
use crate::gfx::texture::DepthTexture;
use crate::scene::object::particle::{Particle, ParticleInstance};
use crate::scene::object::plane::Plane;
use crate::scene::world_map::{Tile, WorldMap};
use crate::scene::Scene;
use crate::cfd::config::Config;
use std::env::args;
use polars::prelude::*;
use polars::prelude::Series;
use polars::prelude::DataFrame;
use polars::prelude::Column;
use polars::prelude::CsvWriter;
use polars::prelude::PolarsResult;
use glam::{Mat4, Vec3, Vec4};
use rand::rngs::ThreadRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::process;
use std::path::Path;
use std::time::{Instant, Duration};
use std::cmp::{min, max};
use winit::event::{ElementState, KeyboardInput, WindowEvent};


mod app;
mod cfd;
mod gfx;
mod scene;

struct FluidSense {
    phong_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    camera: Camera<Perspective>,
    camera_controller: FirstPersonController,
    world_map: WorldMap,
    scene: Scene,
    light: Light,
    particle: Particle,
    particle_instance_buffer: VertexBuffer,
    sph: SPH,
    timer: f32,
    counter: i32,
    df: PolarsResult<DataFrame>,
    config: Config,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    config: String,
    #[arg(long, default_value_t = false)]
    headless: bool,
}

impl App for FluidSense {
    fn init(renderer: &mut Renderer) -> Self {
        let args = Args::parse();
        let phong_pipeline = Pipeline::phong(renderer);
        let particle_pipeline = Pipeline::particle(renderer);
        let config = cfd::config::Config::new(&args.config);
        let mut world_map = WorldMap::new(&config);
        let scene = world_map.build_scene(renderer, &phong_pipeline);
        let (x, z) = scene.user_position();
        let projection = Perspective::new(45.0, renderer.get_aspect_ratio(), 0.1, 1000.0);

        let camera = Camera::new(
            &renderer,
            &phong_pipeline,
            Vec3::new(x, 1.65, z),
            projection,
        );

        let camera_controller = FirstPersonController::new(0.0, 90.0, 4.0, 0.1);
        let light = Light::new(&renderer, &phong_pipeline, camera.position(), Vec3::ONE);
        let particle = Particle::new(renderer);
        let sph = SPH::new(&config);
        let particle_instance_buffer = VertexBuffer::new(renderer, sph.get_particle_instances());

        let mut timer = 0.0;
        let mut counter = 1;

        let count = Column::new("Count".into(), [0]);
        let a1 = Column::new("A1".into(), [22.2 as f32]);
        let a2 = Column::new("A2".into(), [22.2 as f32]);
        let a3 = Column::new("A3".into(), [21.9 as f32]);
        let a4 = Column::new("A4".into(), [21.6 as f32]);
        let a5 = Column::new("A5".into(), [22.0 as f32]);
        let a6 = Column::new("A6".into(), [22.3 as f32]);
        let a7 = Column::new("A7".into(), [22.3 as f32]);
        let a8 = Column::new("A8".into(), [22.4 as f32]);
        let a9 = Column::new("A9".into(), [22.1 as f32]);
        let a10 = Column::new("A10".into(), [22.2 as f32]);
        let a11 = Column::new("A11".into(), [22.4 as f32]);
        let a12 = Column::new("A12".into(), [22.4 as f32]);
        let a13 = Column::new("A13".into(), [22.2 as f32]);
        let a14 = Column::new("A14".into(), [22.2 as f32]);
        let a15 = Column::new("A15".into(), [22.3 as f32]);
        let df =  DataFrame::new(vec![count,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15]);

        Self {
            phong_pipeline,
            particle_pipeline,
            camera,
            camera_controller,
            scene,
            world_map,
            light,
            particle,
            particle_instance_buffer,
            sph,
            timer,
            counter,
            df,
            config,
        }
    }

    fn keyboard_input(&mut self, input: KeyboardInput) {
        self.camera_controller.keyboard_input(input);
    }

    fn mouse_movement(&mut self, dx: f32, dy: f32) {
        self.camera_controller.mouse_movement(dx, dy);
    }

    fn update(&mut self, dt: Duration) {
        self.camera_controller.update(&mut self.camera, dt);
        self.light.set_position(self.camera.position());
        self.sph.step(0.001);
        self.sph.check_particles(&self.world_map);

        self.world_map
            .get_actuators()
            .iter_mut()
            .for_each(|(_, actuator)| match actuator.emit_particle(&dt) {
                None => {}
                Some(particle) => {
                    self.sph.add_particle(particle);
                }
            });
        
        //let config = cfd::config::Config::new(&args.config);

        if self.counter < 214
       {     
            self.timer += dt.as_secs_f32();
            if self.timer > 0.28
            {
                let mut nova_linha = df!(
                    "Count" => &[self.counter],
                    "A1" => &[22.0 as f32],
                    "A2" => &[22.0 as f32],
                    "A3" => &[22.0 as f32],
                    "A4" => &[22.0 as f32],
                    "A5" => &[22.0 as f32],
                    "A6" => &[22.0 as f32],
                    "A7" => &[22.0 as f32],
                    "A8" => &[22.0 as f32],
                    "A9" => &[22.0 as f32],
                    "A10" => &[22.0 as f32],
                    "A11" => &[22.0 as f32],
                    "A12" => &[22.0 as f32],
                    "A13" => &[22.0 as f32],
                    "A14" => &[22.0 as f32],
                    "A15" => &[22.0 as f32]
                );

                let mut A1:f32 = 22.0;
                let mut A2:f32 = 22.0;
                let mut A3:f32 = 22.0;
                let mut A4:f32 = 22.0;
                let mut A5:f32 = 22.0;
                let mut A6:f32 = 22.0;
                let mut A7:f32 = 22.0;
                let mut A8:f32 = 22.0;
                let mut A9:f32 = 22.0;
                let mut A10:f32 = 22.0;
                let mut A11:f32 = 22.0;
                let mut A12:f32 = 22.0;
                let mut A13:f32 = 22.0;
                let mut A14:f32 = 22.0;
                let mut A15:f32 = 22.0;

        self.sph.get_particles().iter().for_each(|particle| {
            match self.world_map.get_device_in_position(particle.position) {
                None => {}
                Some(label) => {
                    let sensors = self.world_map.get_sensor_by_label(&label);
                        
                        if particle.position.z >= 2.495 && particle.position.z < 3.165
                        {
                            //println!("Sensor A1, A4, A7, A10, A13");
                            if particle.position.y >= 0.665 && particle.position.y < 1.335 //A1
                            {
                                if A1 == 22.0 { A1 = particle.temperature; }
                                else { A1 = (A1 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 1.335 && particle.position.y < 2.005 //A4
                            {
                                if A4 == 22.0 { A4 = particle.temperature; }
                                else { A4 = (A4 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.005 && particle.position.y < 2.675 //A7
                            {
                                if A7 == 22.0 { A7 = particle.temperature; }
                                else { A7 = (A7 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.675 && particle.position.y < 3.345 //A10
                            {
                                if A10 == 22.0 { A10 = particle.temperature; }
                                else { A10 = (A10 + particle.temperature)/2.0; }
                            } 
                            else if particle.position.y >= 3.345 && particle.position.y < 4.015 //A13
                            {
                                if A13 == 22.0 { A13 = particle.temperature; }
                                else { A13 = (A13 + particle.temperature)/2.0; }
                            }      
                        }
                        else if particle.position.z >= 3.165 && particle.position.z < 3.835
                        {
                            //println!("Sensor A3, A5, A8, A11, A14");
                            if particle.position.y >= 0.665 && particle.position.y < 1.335 //A3
                            {
                                if A3 == 22.0 { A3 = particle.temperature; }
                                else { A3 = (A3 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 1.335 && particle.position.y < 2.005 //A5
                            {
                                if A5 == 22.0 { A5 = particle.temperature; }
                                else { A5 = (A5 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.005 && particle.position.y < 2.675 //A8
                            {
                                if A8 == 22.0 { A8 = particle.temperature; }
                                else { A8 = (A8 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.675 && particle.position.y < 3.345 //A11
                            {
                                if A11 == 22.0 { A11 = particle.temperature; }
                                else { A11 = (A11 + particle.temperature)/2.0; }
                            } 
                            else if particle.position.y >= 3.345 && particle.position.y < 4.015 //A14
                            {
                                if A14 == 22.0 { A14 = particle.temperature; }
                                else { A14 = (A14 + particle.temperature)/2.0; }
                            }  
                        }
                        else if particle.position.z >= 3.835 && particle.position.z < 4.505
                        {
                            //println!("Sensor A2, A6, A9, A12, A15");
                            if particle.position.y >= 0.665 && particle.position.y < 1.335 //A2
                            {
                                if A2 == 22.0 { A2 = particle.temperature; }
                                else { A2 = (A2 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 1.335 && particle.position.y < 2.005 //A6
                            {
                                if A6 == 22.0 { A6 = particle.temperature; }
                                else { A6 = (A6 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.005 && particle.position.y < 2.675 //A9
                            {
                                if A9 == 22.0 { A9 = particle.temperature; }
                                else { A9 = (A9 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.675 && particle.position.y < 3.345 //A12
                            {
                                if A12 == 22.0 { A12 = particle.temperature; }
                                else { A12 = (A12 + particle.temperature)/2.0; }
                            } 
                            else if particle.position.y >= 3.345 && particle.position.y < 4.015 //A15
                            {
                                if A15 == 22.0 { A15 = particle.temperature; }
                                else { A15 = (A15 + particle.temperature)/2.0; }
                            }  
                        }
                }
            }
        });
                let tabela = self.df.clone().unwrap();
                let id_count = (self.counter - 1) as usize;
                
                //A1
                    let coluna1 = tabela.column("A1").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A1 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A1 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo A1 :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a1 = Series::new("A1".into(), &[a1_val as f32]);
                //A2
                    let coluna1 = tabela.column("A2").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A2 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A2 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo A2 :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a2 = Series::new("A2".into(), &[a1_val as f32]);
                //A3
                    let coluna1 = tabela.column("A3").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A3 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A3 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo A3 :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a3 = Series::new("A3".into(), &[a1_val as f32]);
                //A4
                    let coluna1 = tabela.column("A4").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A4 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A4 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a4 = Series::new("A4".into(), &[a1_val as f32]);
                //A5
                    let coluna1 = tabela.column("A5").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A5 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A5 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a5 = Series::new("A5".into(), &[a1_val as f32]);
                //A6
                    let coluna1 = tabela.column("A6").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A6 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A6 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a6 = Series::new("A6".into(), &[a1_val as f32]);
                //A7
                    let coluna1 = tabela.column("A7").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A7 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A7 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a7 = Series::new("A7".into(), &[a1_val as f32]);
                //A8
                    let coluna1 = tabela.column("A8").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A8 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A8 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a8 = Series::new("A8".into(), &[a1_val as f32]);
                //A9
                    let coluna1 = tabela.column("A9").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A9 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A9 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a9 = Series::new("A9".into(), &[a1_val as f32]);
                //A10
                    let coluna1 = tabela.column("A10").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A10 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A10 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a10 = Series::new("A10".into(), &[a1_val as f32]);
                //A11
                    let coluna1 = tabela.column("A11").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A11 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A11 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a11 = Series::new("A11".into(), &[a1_val as f32]);
                //A12
                    let coluna1 = tabela.column("A12").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A12 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A12 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a12 = Series::new("A12".into(), &[a1_val as f32]);
                //A13
                    let coluna1 = tabela.column("A13").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A13 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A13 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a13 = Series::new("A13".into(), &[a1_val as f32]);
                //A14
                    let coluna1 = tabela.column("A14").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A14 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A14 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a14 = Series::new("A14".into(), &[a1_val as f32]);
                //A15
                    let coluna1 = tabela.column("A15").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A15 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - self.config.get_simulation_config().thermal_conductivity) * val_f32) + (A15 * self.config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a15 = Series::new("A15".into(), &[a1_val as f32]);
                

                let mut df_teste = nova_linha.unwrap();
                df_teste.with_column(nova_a1);
                df_teste.with_column(nova_a2);
                df_teste.with_column(nova_a3);
                df_teste.with_column(nova_a4);
                df_teste.with_column(nova_a5);
                df_teste.with_column(nova_a6);
                df_teste.with_column(nova_a7);
                df_teste.with_column(nova_a8);
                df_teste.with_column(nova_a9);
                df_teste.with_column(nova_a10);
                df_teste.with_column(nova_a11);
                df_teste.with_column(nova_a12);
                df_teste.with_column(nova_a13);
                df_teste.with_column(nova_a14);
                df_teste.with_column(nova_a15);

                //println!("\n Nova Linha:\n{:?}", df_teste);
                
                
                self.counter +=1;
                println!("Conter: {}", self.counter);
                let newdf = self.df.clone().unwrap();
                self.df = newdf.vstack(&df_teste);
                //println!("\nDataFrame atualizado:\n{:?}", self.df);
                self.timer = 0.0;
            }
        }
        else if self.counter == 214
        {
            /*let mut arquivo = File::create("saida.csv")?;
            let mut df_export = self.df.unwrap();
            CsvWriter::new(arquivo).has_header(true).with_delimiter(',').finish(&df_export)?;*/
            let mut df_export = self.df.clone().unwrap();
            
            let mut counter_name = 1;
            let mut ver = 0;
            let mut file_name : String  = ("saida.csv").to_string(); 
            /*while ver == 0 {
                file_name = (("saida").to_string() + &counter_name.to_string() + (".csv"));
                
                if !Path::new(&file_name).exists()
                { ver = 1; }
                else  { counter_name += 1; }

            }*/
            let mut file = std::fs::File::create(file_name).unwrap();

            CsvWriter::new(&mut file).finish(&mut df_export).unwrap();
            process::exit(1);
            self.counter +=1;
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.camera.resize(width, height);
    }

    fn render<'a>(&'a mut self, renderer: &Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.phong_pipeline);
        self.camera.update(&renderer, render_pass);
        self.light.update(&renderer, render_pass);
        self.scene.draw_mesh(render_pass);
        render_pass.set_pipeline(&self.particle_pipeline);
        self.particle_instance_buffer
            .update(renderer, self.sph.get_particle_instances());
        self.particle
            .draw_instanced(render_pass, &self.particle_instance_buffer);
    }
}

fn run_headless() {
    let args = Args::parse();
    let config = cfd::config::Config::new(&args.config);
    let mut sph = SPH::new(&config);
    let mut world_map = WorldMap::new(&config);
    let mut timer: f32 = 0.0;
    let mut counter: i32 = 1;
    let count = Column::new("Count".into(), [0]);
    let a1 = Column::new("A1".into(), [22.2 as f32]);
    let a2 = Column::new("A2".into(), [22.2 as f32]);
    let a3 = Column::new("A3".into(), [21.9 as f32]);
    let a4 = Column::new("A4".into(), [21.6 as f32]);
    let a5 = Column::new("A5".into(), [22.0 as f32]);
    let a6 = Column::new("A6".into(), [22.3 as f32]);
    let a7 = Column::new("A7".into(), [22.3 as f32]);
    let a8 = Column::new("A8".into(), [22.4 as f32]);
    let a9 = Column::new("A9".into(), [22.1 as f32]);
    let a10 = Column::new("A10".into(), [22.2 as f32]);
    let a11 = Column::new("A11".into(), [22.4 as f32]);
    let a12 = Column::new("A12".into(), [22.4 as f32]);
    let a13 = Column::new("A13".into(), [22.2 as f32]);
    let a14 = Column::new("A14".into(), [22.2 as f32]);
    let a15 = Column::new("A15".into(), [22.3 as f32]);
    let mut df =  DataFrame::new(vec![count,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15]);

    loop {
        sph.step(0.001);
        sph.check_particles(&world_map);

        let dt = Duration::from_secs_f64(0.016);

        world_map
            .get_actuators()
            .iter_mut()
            .for_each(|(_, actuator)| match actuator.emit_particle(&dt) {
                None => {}
                Some(particle) => {
                    sph.add_particle(particle);
                }
            });

        if counter < 214
       {     
            timer += dt.as_secs_f32();
            if timer > 0.28
            {
                let mut nova_linha = df!(
                    "Count" => &[counter],
                    "A1" => &[22.0 as f32],
                    "A2" => &[22.0 as f32],
                    "A3" => &[22.0 as f32],
                    "A4" => &[22.0 as f32],
                    "A5" => &[22.0 as f32],
                    "A6" => &[22.0 as f32],
                    "A7" => &[22.0 as f32],
                    "A8" => &[22.0 as f32],
                    "A9" => &[22.0 as f32],
                    "A10" => &[22.0 as f32],
                    "A11" => &[22.0 as f32],
                    "A12" => &[22.0 as f32],
                    "A13" => &[22.0 as f32],
                    "A14" => &[22.0 as f32],
                    "A15" => &[22.0 as f32]
                );

                let mut A1:f32 = 22.0;
                let mut A2:f32 = 22.0;
                let mut A3:f32 = 22.0;
                let mut A4:f32 = 22.0;
                let mut A5:f32 = 22.0;
                let mut A6:f32 = 22.0;
                let mut A7:f32 = 22.0;
                let mut A8:f32 = 22.0;
                let mut A9:f32 = 22.0;
                let mut A10:f32 = 22.0;
                let mut A11:f32 = 22.0;
                let mut A12:f32 = 22.0;
                let mut A13:f32 = 22.0;
                let mut A14:f32 = 22.0;
                let mut A15:f32 = 22.0;

        sph.get_particles().iter().for_each(|particle| {
            match world_map.get_device_in_position(particle.position) {
                None => {}
                Some(label) => {
                    let sensors = world_map.get_sensor_by_label(&label);
                        
                        if particle.position.z >= 2.495 && particle.position.z < 3.165
                        {
                            //println!("Sensor A1, A4, A7, A10, A13");
                            if particle.position.y >= 0.665 && particle.position.y < 1.335 //A1
                            {
                                if A1 == 22.0 { A1 = particle.temperature; }
                                else { A1 = (A1 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 1.335 && particle.position.y < 2.005 //A4
                            {
                                if A4 == 22.0 { A4 = particle.temperature; }
                                else { A4 = (A4 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.005 && particle.position.y < 2.675 //A7
                            {
                                if A7 == 22.0 { A7 = particle.temperature; }
                                else { A7 = (A7 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.675 && particle.position.y < 3.345 //A10
                            {
                                if A10 == 22.0 { A10 = particle.temperature; }
                                else { A10 = (A10 + particle.temperature)/2.0; }
                            } 
                            else if particle.position.y >= 3.345 && particle.position.y < 4.015 //A13
                            {
                                if A13 == 22.0 { A13 = particle.temperature; }
                                else { A13 = (A13 + particle.temperature)/2.0; }
                            }      
                        }
                        else if particle.position.z >= 3.165 && particle.position.z < 3.835
                        {
                            //println!("Sensor A3, A5, A8, A11, A14");
                            if particle.position.y >= 0.665 && particle.position.y < 1.335 //A3
                            {
                                if A3 == 22.0 { A3 = particle.temperature; }
                                else { A3 = (A3 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 1.335 && particle.position.y < 2.005 //A5
                            {
                                if A5 == 22.0 { A5 = particle.temperature; }
                                else { A5 = (A5 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.005 && particle.position.y < 2.675 //A8
                            {
                                if A8 == 22.0 { A8 = particle.temperature; }
                                else { A8 = (A8 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.675 && particle.position.y < 3.345 //A11
                            {
                                if A11 == 22.0 { A11 = particle.temperature; }
                                else { A11 = (A11 + particle.temperature)/2.0; }
                            } 
                            else if particle.position.y >= 3.345 && particle.position.y < 4.015 //A14
                            {
                                if A14 == 22.0 { A14 = particle.temperature; }
                                else { A14 = (A14 + particle.temperature)/2.0; }
                            }  
                        }
                        else if particle.position.z >= 3.835 && particle.position.z < 4.505
                        {
                            //println!("Sensor A2, A6, A9, A12, A15");
                            if particle.position.y >= 0.665 && particle.position.y < 1.335 //A2
                            {
                                if A2 == 22.0 { A2 = particle.temperature; }
                                else { A2 = (A2 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 1.335 && particle.position.y < 2.005 //A6
                            {
                                if A6 == 22.0 { A6 = particle.temperature; }
                                else { A6 = (A6 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.005 && particle.position.y < 2.675 //A9
                            {
                                if A9 == 22.0 { A9 = particle.temperature; }
                                else { A9 = (A9 + particle.temperature)/2.0; }
                            }
                            else if particle.position.y >= 2.675 && particle.position.y < 3.345 //A12
                            {
                                if A12 == 22.0 { A12 = particle.temperature; }
                                else { A12 = (A12 + particle.temperature)/2.0; }
                            } 
                            else if particle.position.y >= 3.345 && particle.position.y < 4.015 //A15
                            {
                                if A15 == 22.0 { A15 = particle.temperature; }
                                else { A15 = (A15 + particle.temperature)/2.0; }
                            }  
                        }
                }
            }
        });
                let tabela = df.clone().unwrap();
                let id_count = (counter - 1) as usize;
                
                //A1
                    let coluna1 = tabela.column("A1").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A1 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A1 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo A1 :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a1 = Series::new("A1".into(), &[a1_val as f32]);
                //A2
                    let coluna1 = tabela.column("A2").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A2 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A2 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo A2 :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a2 = Series::new("A2".into(), &[a1_val as f32]);
                //A3
                    let coluna1 = tabela.column("A3").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A3 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A3 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo A3 :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a3 = Series::new("A3".into(), &[a1_val as f32]);
                //A4
                    let coluna1 = tabela.column("A4").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A4 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A4 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a4 = Series::new("A4".into(), &[a1_val as f32]);
                //A5
                    let coluna1 = tabela.column("A5").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A5 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A5 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a5 = Series::new("A5".into(), &[a1_val as f32]);
                //A6
                    let coluna1 = tabela.column("A6").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A6 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A6 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a6 = Series::new("A6".into(), &[a1_val as f32]);
                //A7
                    let coluna1 = tabela.column("A7").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A7 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A7 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a7 = Series::new("A7".into(), &[a1_val as f32]);
                //A8
                    let coluna1 = tabela.column("A8").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A8 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A8 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a8 = Series::new("A8".into(), &[a1_val as f32]);
                //A9
                    let coluna1 = tabela.column("A9").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A9 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A9 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a9 = Series::new("A9".into(), &[a1_val as f32]);
                //A10
                    let coluna1 = tabela.column("A10").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A10 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A10 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a10 = Series::new("A10".into(), &[a1_val as f32]);
                //A11
                    let coluna1 = tabela.column("A11").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A11 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A11 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a11 = Series::new("A11".into(), &[a1_val as f32]);
                //A12
                    let coluna1 = tabela.column("A12").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A12 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A12 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a12 = Series::new("A12".into(), &[a1_val as f32]);
                //A13
                    let coluna1 = tabela.column("A13").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A13 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A13 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a13 = Series::new("A13".into(), &[a1_val as f32]);
                //A14
                    let coluna1 = tabela.column("A14").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A14 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A14 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a14 = Series::new("A14".into(), &[a1_val as f32]);
                //A15
                    let coluna1 = tabela.column("A15").unwrap();
                    let item_coluna1 = coluna1.get(id_count);
                    let aux_item_coluna1 = item_coluna1.unwrap();
                    let opt_f32 = aux_item_coluna1.extract::<f32>();
                    let val_f32: f32 = match opt_f32 { Some(v) => v as f32, None => 0.0 };
                    let mut a1_val = 0 as f32;
                    if A15 != 22.0
                    { a1_val = f32::max(22.0,( (((1.0 - config.get_simulation_config().thermal_conductivity) * val_f32) + (A15 * config.get_simulation_config().thermal_conductivity))));
                   //println!("Teste Calculo :\n{}", a1_val); 
                    }
                    else { a1_val = val_f32; }
                    let nova_a15 = Series::new("A15".into(), &[a1_val as f32]);
                

                let mut df_teste = nova_linha.unwrap();
                df_teste.with_column(nova_a1);
                df_teste.with_column(nova_a2);
                df_teste.with_column(nova_a3);
                df_teste.with_column(nova_a4);
                df_teste.with_column(nova_a5);
                df_teste.with_column(nova_a6);
                df_teste.with_column(nova_a7);
                df_teste.with_column(nova_a8);
                df_teste.with_column(nova_a9);
                df_teste.with_column(nova_a10);
                df_teste.with_column(nova_a11);
                df_teste.with_column(nova_a12);
                df_teste.with_column(nova_a13);
                df_teste.with_column(nova_a14);
                df_teste.with_column(nova_a15);

                //println!("\n Nova Linha:\n{:?}", df_teste);
                
                
                counter +=1;
                println!("Conter: {}", counter);
                let newdf = df.clone().unwrap();
                df = newdf.vstack(&df_teste);
                //println!("\nDataFrame atualizado:\n{:?}", self.df);
                timer = 0.0;
            }
        }
        else if counter == 214
        {
            /*let mut arquivo = File::create("saida.csv")?;
            let mut df_export = self.df.unwrap();
            CsvWriter::new(arquivo).has_header(true).with_delimiter(',').finish(&df_export)?;*/
            let mut df_export = df.clone().unwrap();
            
            let mut counter_name = 1;
            let mut ver = 0;
            let mut file_name : String  = ("saida.csv").to_string(); 
            /*while ver == 0 {
                file_name = (("saida").to_string() + &counter_name.to_string() + (".csv"));
                
                if !Path::new(&file_name).exists()
                { ver = 1; }
                else  { counter_name += 1; }

            }*/
            let mut file = std::fs::File::create(file_name).unwrap();

            CsvWriter::new(&mut file).finish(&mut df_export).unwrap();
            process::exit(1);
            counter +=1;
        }
    }
}

fn main() {
    let args = Args::parse();

    if args.headless {
        run_headless();
    } else {
        pollster::block_on(app::run::<FluidSense>());
    }
}

