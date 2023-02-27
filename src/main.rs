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
use rayon::prelude::{IntoParallelRefIterator, IntoParallelIterator, ParallelIterator};

use crate::atom::Atom;
use crate::atom::species::Species;
use crate::rule::Rule;

mod atom;
mod rule;

type F = f32;

#[allow(dead_code)]
enum SpawnArea
{
    Rectangle,
    Ellipse
}

const POPULATION: usize = 300*SPECIES_COUNT;
const INITIAL_WINDOW_SIZE: [f64; 2] = [640.0, 480.0];
const G_MUL: F = 2.5;
const G_BIAS: F = 0.0;
const G_BIAS_MUL: F = 1.0;
const VISCOUSITY: F = 1.0;
const SUCK: F = 1.0;
const SPAWN_AREA: SpawnArea = SpawnArea::Ellipse;
const SIMULTANEOUS_ENFORCEMENTS: usize = 64;
const MAX_G_ORDER: u8 = 1;
const RFRAMES: usize = 1;

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

    let rules: [Vec<Rule>; SPECIES_COUNT] = {
        let g_variance: F = G_MUL;
        let g_weightset_count = Uniform::from(5..32).sample(rng);
        let g_weightset: Vec<F> = (0..g_weightset_count)
            .map(|_| random_gauss(rng)*g_variance)
            .collect();
        //let g_duality = random_gauss(rng)*g_variance;

        array_init(|i| {
            //let g_duality = if i < SPECIES_COUNT/2 {g_duality} else {-g_duality};

            let b = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)]*G_BIAS_MUL;
            let s = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)] + 1.0;

            (1..=Uniform::from(1..=MAX_G_ORDER).sample(rng))
                .map(|power| [power]
                        .iter()
                        .map(|power| 
                            {
                                let rng = &mut rand::thread_rng();

                                let w = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)] + 1.0;
                                let m = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];

                                Rule::GravitySelf( 
                                    random_gauss(rng)*g_variance + ((random_gauss(rng) + 1.0)*g_variance + w)*random_gauss(rng)*m*s + b + G_BIAS,
                                    *power
                                )
                            }
                        )
                        .chain((0..SPECIES_COUNT)
                            .filter_map(|j| if i != j {Some(&ATOM_SPECIES[j])} else {None})
                            .filter_map(|from|
                                {
                                    if Uniform::from(0.0..1.0).sample(rng) <= 0.4
                                    {
                                        let w = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];
                                        let m = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];
            
                                        Some(
                                            Rule::Gravity(
                                                from,
                                                random_gauss(rng)*g_variance + ((random_gauss(rng) + 1.0)*g_variance + w)*random_gauss(rng)*m*s + b + G_BIAS,
                                                power
                                            )
                                        )
                                    }
                                    else
                                    {
                                        None
                                    }
                                }
                            )
                        )
                        .collect()
                )
                .reduce(|accum, vec| [accum, vec].concat())
                .unwrap_or_default()
        })
    };

    let population = {
        let mut population: [usize; SPECIES_COUNT] = [0; SPECIES_COUNT];
        (0..POPULATION)
            .map(|_| Uniform::from(0..SPECIES_COUNT).sample(rng))
            .for_each(|i| population[i] += 1);
        population
    };

    let atoms: Arc<[Vec<RwLock<Atom>>; SPECIES_COUNT]> = Arc::new(ATOM_SPECIES
        .map(|species| (0..population[species.id as usize])
            .map(|_|
            {
                let center = [world_size[0]*0.5, world_size[1]*0.5];
                RwLock::new(Atom {
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
                })
            })
            .collect()
        ));

    let cycle_rules_sequence: Vec<(usize, Rule)> = ATOM_SPECIES.iter()
        .flat_map(|species| rules[species.index()]
            .iter()
            .map(|rule| (species.index(), *rule))
        )
        .collect();
    
    let cycle_rules: Arc<Mutex<Vec<(usize, Rule)>>> = Arc::new(Mutex::new(cycle_rules_sequence.clone()));
    
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
            let mut cycle_rules = cycle_rules.lock().unwrap();
            while cycle_rules.len() < SIMULTANEOUS_ENFORCEMENTS.min(cycle_rules_sequence.len())
            {
                let mut cycle_rules_sequence = cycle_rules_sequence.clone();
                cycle_rules_sequence.sort_by(|_, _| match Uniform::from(0..2).sample(&mut rand::thread_rng()) == 1
                {
                    true => Ordering::Greater,
                    _ => Ordering::Less
                });
                cycle_rules.append(&mut cycle_rules_sequence);
            }
        }
        //ENFORCE RULES
        (0..SIMULTANEOUS_ENFORCEMENTS.min(cycle_rules_sequence.len()))
            .map(|_| cycle_rules.clone())
            .collect::<Vec<Arc<Mutex<Vec<(usize, Rule)>>>>>()
            .into_par_iter()
            .map(|cycle_rules| loop
            {
                match {
                    let cycle_rules = cycle_rules.clone();
                    loop
                    {
                        match cycle_rules.lock()
                        {
                            Ok(mut cycle_rules) => break cycle_rules.pop(),
                            Err(_) => ()
                        }
                    }
                }
                {
                    Some((index, rule)) => {
                        //let species = ATOM_SPECIES[index];
                        break (index, match rule
                        {
                            Rule::Gravity(from_species, g, power) => {
                                let from = &atoms[from_species.id as usize];
        
                                (&atoms[index]).iter()
                                    //.filter(|atom| atom.is_species(species))
                                    .enumerate()
                                    .map(|(i, atom)| {
                                        atom.read()
                                            .ok()
                                            .map(|atom| {
                                                atom.gravity_from_group(
                                                    from.iter()
                                                        .enumerate()
                                                        .filter(|(j, _)| from_species.id as usize != index || i != *j)
                                                        .map(|(_, from)| from),
                                                    world_size,
                                                    g,
                                                    power
                                                )
                                            })
                                            .unwrap_or([0.0, 0.0])
                                    }).collect()
                            },
                            Rule::GravitySelf(g, power) => {
                                let atoms_group = &atoms[index];
                                    
                                Atom::gravity_all(atoms_group, world_size, g, power)
                            }
                        })
                    },
                    None => println!("Empty")
                }
            }).collect::<Vec<(usize, Vec<[F; 2]>)>>()
            .into_iter()
            .for_each(|(index, gravity)| {
                for (mut atom, gravity) in atoms[index].iter()
                    .zip(gravity.into_iter())
                    .filter_map(|(atom, gravity)| atom.write().ok().map(|atom| (atom, gravity)))
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
                        let atoms_group = &atoms[index];
                        atoms_group
                            .iter()
                            .for_each(|atom| 
                                {
                                    let mut atom = atom.write().unwrap();
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
                            let atom = atom.read().unwrap();
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
                    let atoms_group = &atoms[index];
                    atoms_group
                        .iter()
                        .for_each(|atom| 
                            {
                                let mut atom = atom.write().unwrap();
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