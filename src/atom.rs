use std::sync::{RwLock, RwLockReadGuard};

use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

pub mod species;

type F = f32;

const CLOSE_RANGE_REPULSION: F = -50.0;
const CLOSE_RANGE_DISTANCE: F = 60.0;
const CLOSE_RANGE_DISTANCE_INV: F = 1.0/CLOSE_RANGE_DISTANCE;

#[derive(Clone, Copy)]
pub struct Atom
{
    pub pos: [F; 2],
    pub vel: [F; 2],
    pub acc: [F; 2],
    pub species: u8
}

impl Atom
{
    pub fn update_movement(&mut self, dt: F, boundry: [[F; 2]; 2], brakes: F)
    {
        self.update_movement_dim::<0>(dt, boundry, brakes);
        self.update_movement_dim::<1>(dt, boundry, brakes)
    }

    pub fn update_movement_dim<const I: usize>(&mut self, dt: F, boundry: [[F; 2]; 2], brakes: F)
    {
        let min = boundry[0][I];
        let max = boundry[1][I];
        if if self.vel[I] < 0.0 {self.pos[I] <= min} else {self.pos[I] >= max}
        {
            self.vel[I] = -self.vel[I]
        }
        self.vel[I] = (self.vel[I] + self.acc[I]*dt)*brakes;
        self.pos[I] = (self.pos[I] + self.vel[I]*dt).max(min).min(max);
        /*self.pos[I] = self.pos[I] + self.vel[I]*dt;
        if self.pos[I] < boundry[0][I]
        {
            self.pos[I] += boundry[1][I] - boundry[0][I]
        }
        if self.pos[I] > boundry[1][I]
        {
            self.pos[I] -= boundry[1][I] - boundry[0][I]
        }*/
    }

    pub fn reset_acc(&mut self)
    {
        self.acc = [0.0, 0.0];
    }

    pub fn apply_force(&mut self, force: [F; 2])
    {
        self.acc[0] += force[0];
        self.acc[1] += force[1];

        /*self.pos[0] += self.vel[0]*dt;
        self.pos[1] += self.vel[1]*dt;*/
    }

    /*pub fn dist_from(&self, pos: [f64; 2]) -> [f64; 2]
    {
        [
            self.pos[0] - pos[0],
            self.pos[1] - pos[1]
        ]
    }*/

    pub fn dist_to(&self, pos: [F; 2], _world_size: [F; 2]) -> [F; 2]
    {
        let d = [
            pos[0] - self.pos[0],
            pos[1] - self.pos[1]
        ];
        /*for i in 0..2
        {
            if d[i] > world_size[i]*0.5
            {
                d[i] -= world_size[i]
            }
            if d[i] < -world_size[i]*0.5
            {
                d[i] += world_size[i]
            }
        };*/
        d
    }

    fn gravity_from(&self, pos: [F; 2], world_size: [F; 2], g: F, power: u8) -> [F; 2]
    {
        let d = self.dist_to(pos, world_size);
        if (d[0] == 0.0 || d[0] == -0.0) && (d[1] == 0.0 || d[1] == -0.0)
        {
            return [0.0, 0.0]
        }
        let d_abs_inv = d[0].hypot(d[1]).recip();
        let g = if g > CLOSE_RANGE_REPULSION && d_abs_inv > CLOSE_RANGE_DISTANCE_INV
        {
            let m = d_abs_inv.recip()/CLOSE_RANGE_DISTANCE;
            g*m + CLOSE_RANGE_REPULSION*(1.0 - m)
        }
        else
        {
            g
        };
        let f_abs = g*d_abs_inv.powi(power as i32 + 1);

        [
            f_abs*d[0],
            f_abs*d[1]
        ]
    }

    pub fn gravity_all(atoms: &[RwLock<Self>], world_size: [F; 2], g: F, power: u8) -> Vec<[F; 2]>
    {
        let mut force = vec![[0.0, 0.0]; atoms.len()];

        let atoms: Vec<(usize, RwLockReadGuard<Atom>)> = atoms.iter()
            .enumerate()
            .filter_map(|(i, atom)| atom.read().ok().map(|atom| (i, atom)))
            .collect();
        atoms
            .iter()
            .for_each(|(i, atom)| (&atoms[0..*i])
                .iter()
                .for_each(|(j, from)|
                {
                    let g = atom.gravity_from(from.pos, world_size, g, power);
                    
                    force[*i][0] += g[0];
                    force[*i][1] += g[1];

                    force[*j][0] -= g[0];
                    force[*j][1] -= g[1];
                })
            );
        
        force
    }

    pub fn average_pos_and_count<'a, I: Iterator<Item = &'a RwLock<Atom>>>(atoms: I) -> (usize, [F; 2])
    {
        atoms
            .filter_map(|atom| atom.read().ok())
            .map(|atom| (1, atom.pos))
            .reduce(|(n, accum), (i, pos)| (n + i, [
                accum[0] + pos[0],
                accum[1] + pos[1]
            ]))
            .map(|(n, pos)| (n, [pos[0]/n as F, pos[1]/n as F]))
            .unwrap_or((0, [0.0, 0.0]))
    }

    pub fn gravity_from_group<'a, I: Iterator<Item = &'a RwLock<Atom>>>(self, from: I, world_size: [F; 2], g: F, power: u8) -> [F; 2]
    {
        let (count, pos) = Self::average_pos_and_count(from);
        self.gravity_from(pos, world_size, g*count as F, power)
    }
}