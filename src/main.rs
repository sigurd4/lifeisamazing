#![windows_subsystem = "windows"]

use std::cmp::Ordering;
use std::f32::consts::TAU;
use std::ops::Mul;
use std::slice::Iter;
use array_init::array_init;

use piston_window::{PistonWindow, WindowSettings};
use rand::distributions::{Uniform, Standard};
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;

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

const POPULATION: usize = 500*SPECIES_COUNT;
const INITIAL_WINDOW_SIZE: [f64; 2] = [640.0, 480.0];
const DT: F = 0.00003/RFRAMES as F;
const G_MUL: F = 450.0;
const G_BIAS: F = -0.0;
const G_BIAS_MUL: F = 40.0;
const VISCOUSITY: F = 600.0;
const SPAWN_AREA: SpawnArea = SpawnArea::Ellipse;
const SIMULTANEOUS_ENFORCEMENTS: usize = 16;
const MAX_G_ORDER: u8 = 2;
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
            let s = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];

            let w = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)] + 1.0;
            let m = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];

            (1..Uniform::from(2..(MAX_G_ORDER + 1)).sample(rng))
                .map(|power| [power]
                        .iter()
                        .map(|power| 
                            {
                                let rng = &mut rand::thread_rng();
                                Rule::GravitySelf( 
                                    random_gauss(rng)*g_variance + (random_gauss(rng)*g_variance + w)*random_gauss(rng)*m*s + b + G_BIAS,
                                    *power
                                )
                            }
                        )
                        .chain((0..SPECIES_COUNT)
                            .filter_map(|j| if i != j {Some(&ATOM_SPECIES[j])} else {None})
                            .filter_map(|from|
                                {
                                    if Uniform::from(0..1).sample(rng) == 0
                                    {
                                        let w = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];
                                        let m = g_weightset[Uniform::from(0..g_weightset_count).sample(rng)];
            
                                        Some(
                                            Rule::Gravity(
                                                from,
                                                random_gauss(rng)*g_variance + (random_gauss(rng)*g_variance + w)*random_gauss(rng)*m*s + b + G_BIAS,
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

    let mut cycle_rules_sequence: Vec<(usize, Rule)> = ATOM_SPECIES.iter()
        .flat_map(|species| rules[species.index()]
            .iter()
            .map(|rule| (species.index(), *rule))
        )
        .collect();
    
    let mut cycle_rules: Iter<(usize, Rule)> = [].iter();
    
    let brakes = (-VISCOUSITY*DT).exp();

    while let Some(event) = window.next()
    {
        //ENFORCE RULES
        for _ in 0..SIMULTANEOUS_ENFORCEMENTS.min(cycle_rules_sequence.len())
        {
            loop
            {
                match cycle_rules.next().map(|(index, rule)| (*index, *rule))
                {
                    Some((index, rule)) => {
                        //let species = ATOM_SPECIES[index];
                        match rule
                        {
                            Rule::Gravity(from_species, g, power) => {
                                let from: Vec<Atom> = (&atoms[from_species.id as usize])
                                    .iter()
                                    //.filter(|atom| atom.is_species(from_species))
                                    .map(|x| *x)
                                    .collect();
        
                                (&mut atoms[index]).iter_mut()
                                    //.filter(|atom| atom.is_species(species))
                                    .for_each(|atom|
                                        atom.apply_force(
                                            atom.gravity_from_group(
                                                from.iter(),
                                                g,
                                                power
                                            )
                                            .map(|x| x)
                                        )
                                    );
                            },
                            Rule::GravitySelf(g, power) => {
                                let atoms_group = &mut atoms[index];
                                let from: Vec<Atom> = atoms_group
                                    .iter()
                                    //.filter(|atom| atom.is_species(species))
                                    .map(|x| *x)
                                    .collect();
                                    
                                let gravity = Atom::gravity_all(&from, g, power);
        
                                atoms_group.iter_mut()
                                    //.filter(|atom| atom.is_species(species))
                                    .enumerate()
                                    .for_each(|(i, atom)| 
                                        atom.apply_force(gravity[i])
                                    );
                            }
                        };
                        break;
                    },
                    None => {
                        cycle_rules_sequence.sort_by(|_, _| match Uniform::from(0..2).sample(rng) == 1
                        {
                            true => Ordering::Greater,
                            _ => Ordering::Less
                        });
                        cycle_rules = cycle_rules_sequence.iter();
                    }
                }
            }
        }
        
        for _ in 0..RFRAMES
        {
            let boundry = [
                [-world_size[0]*0.5, -world_size[1]*0.5],
                [world_size[0]*0.5, world_size[1]*0.5]
            ];
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
                                    atom.update_movement(DT, boundry, brakes);
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

                        let dt = DT as F;
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