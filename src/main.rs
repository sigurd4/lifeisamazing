#![windows_subsystem = "windows"]

use std::cmp::Ordering;
use std::f32::consts::TAU;
use std::ops::Mul;
use std::slice::Iter;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::SystemTime;
use array_init::array_init;

use piston_window::{PistonWindow, WindowSettings, TimeStamp};
use rand::distributions::{Uniform, Standard};
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand_distr::StandardNormal;
use rayon::prelude::{IntoParallelRefIterator, IntoParallelIterator, ParallelIterator};

use crate::atom::Atom;
use crate::atom::species::Species;

mod atom;
mod fast_invsqrt;

type F = f32;

#[allow(dead_code)]
enum SpawnArea
{
    Rectangle,
    Ellipse
}

const POPULATION: usize = 300*SPECIES_COUNT;
const INITIAL_WINDOW_SIZE: [f64; 2] = [640.0, 480.0];
const SPAWN_AREA: SpawnArea = SpawnArea::Ellipse;
const WORLD_WRAPPING: bool = false;

const MIN_G_ORDER: u8 = 1;
const MAX_G_ORDER: u8 = 1;
const G_ORDER_COUNT: usize = 1 + MAX_G_ORDER as usize - MIN_G_ORDER as usize;
const G_VARIANCE: F = 200.0;
const G_MEAN: F = 0.0;

const VISCOUSITY: F = 3.0;
const SUCK: F = 0.03;

const RFRAMES: usize = 1;
const SIMULTANEOUS_ENFORCEMENTS: usize = 64;

const RED: Species = Species {
    id: 0,
    size: 5.0,
    color: [1.0, 0.0, 0.4, 1.0]
};
const GREEN: Species = Species {
    id: 1,
    size: 5.0,
    color: [0.5, 1.0, 0.1, 1.0]
};
const BLUE: Species = Species {
    id: 2,
    size: 5.0,
    color: [0.2, 0.2, 0.9, 1.0]
};
const YELLOW: Species = Species {
    id: 3,
    size: 5.0,
    color: [0.8, 0.8, 0.0, 1.0]
};
const CYAN: Species = Species {
    id: 4,
    size: 5.0,
    color: [0.0, 0.8, 1.0, 1.0]
};
const PURPLE: Species = Species {
    id: 5,
    size: 5.0,
    color: [0.5, 0.4, 0.9, 1.0]
};
const ORANGE: Species = Species {
    id: 6,
    size: 5.0,
    color: [1.0, 0.5, 0.0, 1.0]
};
const WHITE: Species = Species {
    id: 7,
    size: 2.0,
    color: [1.0, 1.0, 1.0, 1.0]
};
const BLACK: Species = Species {
    id: 8,
    size: 0.0,
    color: [0.0, 0.0, 0.0, 0.0]
};

const SPECIES_COUNT: usize = 9;
const ATOM_SPECIES: [&Species; SPECIES_COUNT] = [
    &RED,
    &GREEN,
    &BLUE,
    &YELLOW,
    &CYAN,
    &PURPLE,
    &ORANGE,
    &WHITE,
    &BLACK
];

fn main() {
    let mut window: PistonWindow = WindowSettings::new("lifeisamazing", INITIAL_WINDOW_SIZE)
        .exit_on_esc(true)
        .resizable(true)
        .build()
        .unwrap();

    let rng = &mut rand::thread_rng();
    
    let mut world_size: [F; 2] = [INITIAL_WINDOW_SIZE[0] as F, INITIAL_WINDOW_SIZE[1] as F];

    let force_matrix: [[[F; SPECIES_COUNT]; SPECIES_COUNT]; G_ORDER_COUNT] = {
        array_init(|_order_rel| {
            //let _order = MIN_G_ORDER + order_rel as u8;
            array_init(|_atom| array_init(|_from| <StandardNormal as Distribution<F>>::sample(&StandardNormal, rng)*G_VARIANCE + G_MEAN))
        })
    };

    let population = {
        let mut population: [usize; SPECIES_COUNT] = [0; SPECIES_COUNT];
        (0..POPULATION)
            .map(|_| Uniform::from(0..SPECIES_COUNT).sample(rng))
            .for_each(|i| population[i] += 1);
        population
    };

    let mut atoms: [Vec<Atom>; SPECIES_COUNT] = ATOM_SPECIES
        .map(|species| (0..population[species.id as usize])
            .map(|_|
            {
                let center = [world_size[0]*0.5, world_size[1]*0.5];
                Atom {
                    pos: match SPAWN_AREA
                    {
                        SpawnArea::Rectangle => [
                            Uniform::from(0.0..world_size[0]).sample(rng) - center[0],
                            Uniform::from(0.0..world_size[1]).sample(rng) - center[1]
                        ],
                        SpawnArea::Ellipse => {
                            let theta = Uniform::from(0.0..TAU).sample(rng);
                            let r = Uniform::from(0.0..1.0).sample(rng);
                            [
                                center[0]*r*theta.cos(),
                                center[1]*r*theta.sin()
                            ]
                        }
                    },
                    vel: [0.0, 0.0],
                    acc: [0.0, 0.0],
                    species: species.id,
                }
            })
            .collect()
        );

    let cycle_rules_sequence: [(u8, u8, u8); G_ORDER_COUNT*SPECIES_COUNT*SPECIES_COUNT] = {
        array_init(|i| ((i/(SPECIES_COUNT*SPECIES_COUNT)) as u8 + MIN_G_ORDER, ((i/SPECIES_COUNT)%SPECIES_COUNT) as u8, (i%SPECIES_COUNT) as u8))
    };
    
    let mut cycle_rules: Vec<(u8, u8, u8)> = vec![];
    
    let mut frame_time = SystemTime::now();

    while let Some(event) = window.next()
    {
        let dt = {
            let now = SystemTime::now();
            let dt = now.duration_since(frame_time).unwrap().as_secs_f32();
            frame_time = now;
            dt
        };

        let brakes = (-VISCOUSITY*dt).exp();

        let boundry = [
            [-world_size[0]*0.5, -world_size[1]*0.5],
            [world_size[0]*0.5, world_size[1]*0.5]
        ];

        {
            while cycle_rules.len() < SIMULTANEOUS_ENFORCEMENTS.min(cycle_rules_sequence.len())
            {
                let mut cycle_rules_sequence = cycle_rules_sequence.to_vec();
                cycle_rules_sequence.sort_by(|_, _| match Uniform::from(0..2).sample(&mut rand::thread_rng()) == 1
                {
                    true => Ordering::Greater,
                    _ => Ordering::Less
                });
                cycle_rules.append(&mut cycle_rules_sequence);
            }
        }
        //ENFORCE RULES
        cycle_rules.drain(0..SIMULTANEOUS_ENFORCEMENTS.min(cycle_rules_sequence.len()))
            .collect::<Vec<(u8, u8, u8)>>()
            .into_par_iter()
            .map(|(power, index, from_species)| {
                let g = {
                    let i = (power - MIN_G_ORDER) as usize;
                    let j = index as usize;
                    let k = from_species as usize;
                    force_matrix[i][j][k]
                };
                //let g = force_matrix[(power - MIN_G_ORDER) as usize][index as usize][from_species as usize];
                
                (index, if index != from_species
                {
                    let from = &atoms[from_species as usize];
                    
                    (&atoms[index as usize]).iter()
                        .map(|atom| {
                            atom.gravity_from_group(
                                from.iter(),
                                world_size,
                                g,
                                power
                            )
                        }).collect()
                }
                else
                {
                    let atoms_group = &atoms[index as usize];
                        
                    Atom::gravity_all(atoms_group, world_size, g, power)
                })
            }).collect::<Vec<(u8, Vec<[F; 2]>)>>()
            .into_iter()
            .for_each(|(index, gravity)| {
                for (atom, gravity) in atoms[index as usize].iter_mut()
                    .zip(gravity.into_iter())
                {
                    let pos = atom.pos;
                    atom.apply_force([gravity[0] - pos[0]*SUCK, gravity[1] - pos[1]*SUCK])
                }

            });
        
        for _ in 0..RFRAMES
        {
            //UPDATE MOVEMENT
            ATOM_SPECIES.iter()
                .for_each(|species|
                    {
                        let index = species.index();
                        let atoms_group = &mut atoms[index];
                        atoms_group
                            .iter_mut()
                            .for_each(|atom| 
                                {
                                    atom.update_movement(dt, boundry, brakes);
                                }
                            );
                    });

            //RENDER
            window.draw_2d(&event, |context, graphics, _device| {
                match context.viewport
                {
                    Some(viewport) => {
                        graphics::clear([0.0, 0.0, 0.0, 1.0], graphics);

                        let window_size = viewport.window_size;
                        let transform = context.transform;

                        world_size = [window_size[0] as F, window_size[1] as F];
                        let world_center = scale(world_size, 0.5);

                        for atom in atoms
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| ATOM_SPECIES[*i].size != 0.0)
                            .flat_map(|(_, atoms_group)| atoms_group)
                        {
                            let draw_pos: [f64; 2] = [
                                (atom.pos[0] + atom.vel[0]*dt as F + world_center[0]) as f64,
                                (atom.pos[1] + atom.vel[1]*dt as F + world_center[1]) as f64
                            ];
                            let species = ATOM_SPECIES[atom.species as usize];

                            graphics::rectangle(species.color,
                                graphics::rectangle::centered([
                                    draw_pos[0],
                                    draw_pos[1],
                                    species.size*0.5,
                                    species.size*0.5
                                ]),
                                transform,
                                graphics
                            );
                            
                            /*let angle0 = atom.vel[1].atan2(atom.vel[0]);
                            let angle = angle0 + 0.25*PI;

                            let cos = angle.cos() as f64*species.size;
                            let sin = angle.sin() as f64*species.size;

                            graphics::polygon(species.color,
                                &[
                                    [cos + draw_pos[0], sin + draw_pos[1]],
                                    [-sin + draw_pos[0], cos + draw_pos[1]],
                                    [-cos + draw_pos[0], -sin + draw_pos[1]],
                                    [sin + draw_pos[0], -cos + draw_pos[1]]
                                    /*[angle0.cos() as f64*species.size + draw_pos[0], angle0.sin() as f64*species.size + draw_pos[1]],
                                    [-sin + draw_pos[0], cos + draw_pos[1]],
                                    [-cos + draw_pos[0], -sin + draw_pos[1]]*/
                                ],
                                transform,
                                graphics
                            );*/
                        }
                    },
                    _ => ()
                }
            });
        }

        //RESET ACCELERATION
        ATOM_SPECIES.iter()
            .for_each(|species|
                {
                    let index = species.index();
                    let atoms_group = &mut atoms[index];
                    atoms_group
                        .iter_mut()
                        .for_each(|atom| 
                            {
                                atom.reset_acc();
                            }
                        );
                });
    }
}

fn random_gauss(rng: &mut ThreadRng) -> F
{
    Uniform::from(-1.0..1.0).sample(rng)
    //<Standard as Distribution<F>>::sample(&Standard, rng)*PARAMETER_VARIANCE
}

fn scale<F1, F2>(window_size: [F1; 2], world_scale: F2) -> [<F1 as Mul<F2>>::Output; 2]
where F1: Mul<F2> + Copy, F2: Copy
{
    [window_size[0]*world_scale, window_size[1]*world_scale]
}

fn scale64(window_size: [f64; 2], world_scale: F) -> [F; 2]
{
    [window_size[0] as F*world_scale, window_size[1] as F*world_scale]
}