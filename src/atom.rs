use std::sync::{RwLock, RwLockReadGuard};

use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

use crate::{MIN_G_ORDER, WORLD_WRAPPING};
use crate::fast_invsqrt::FastInvSqrt;

pub mod species;

type F = f32;

const CLOSE_RANGE_REPULSION: F = -100.0;
const CLOSE_RANGE_DISTANCE: F = 80.0;
const CLOSE_RANGE_DISTANCE_SQR: F = CLOSE_RANGE_DISTANCE*CLOSE_RANGE_DISTANCE;
const CLOSE_RANGE_DISTANCE_INV_SQR: F = 1.0/CLOSE_RANGE_DISTANCE_SQR;

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
        if WORLD_WRAPPING
        {
            self.pos[I] = self.pos[I] + self.vel[I]*dt;
            if self.pos[I] < boundry[0][I]
            {
                self.pos[I] += boundry[1][I] - boundry[0][I]
            }
            if self.pos[I] > boundry[1][I]
            {
                self.pos[I] -= boundry[1][I] - boundry[0][I]
            }
        }
        else
        {
            self.pos[I] = (self.pos[I] + self.vel[I]*dt).max(min).min(max);
        }
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

    pub fn dist_to(&self, pos: [F; 2], world_size: [F; 2]) -> [F; 2]
    {
        let mut d = [
            pos[0] - self.pos[0],
            pos[1] - self.pos[1]
        ];
        if WORLD_WRAPPING
        {
            for i in 0..2
            {
                if d[i] > world_size[i]*0.5
                {
                    d[i] -= world_size[i]
                }
                if d[i] < -world_size[i]*0.5
                {
                    d[i] += world_size[i]
                }
            };
        }
        d
    }

    fn gravity_from(&self, pos: [F; 2], world_size: [F; 2], g: F, power: u8) -> [F; 2]
    {
        let d = self.dist_to(pos, world_size);
        if (d[0] == 0.0 || d[0] == -0.0) && (d[1] == 0.0 || d[1] == -0.0)
        {
            return [0.0, 0.0]
        }
        let d_abs2 = d[0]*d[0] + d[1]*d[1];
        let g = if power == MIN_G_ORDER && g > CLOSE_RANGE_REPULSION && d_abs2 < CLOSE_RANGE_DISTANCE_SQR
        {
            let m = d_abs2*CLOSE_RANGE_DISTANCE_INV_SQR;
            g*m + CLOSE_RANGE_REPULSION*(1.0 - m)
            //CLOSE_RANGE_REPULSION
        }
        else
        {
            g
        };
        if g == 0.0
        {
            return [0.0, 0.0]
        }
        let d_abs_inv = d_abs2.fast_invsqrt();
        let mut f_abs = g*d_abs_inv;
        for _ in 1..=power
        {
            f_abs *= d_abs_inv;
        }

        [
            f_abs*d[0],
            f_abs*d[1]
        ]
    }

    pub fn gravity_all(atoms: &[Self], world_size: [F; 2], g: F, power: u8) -> Vec<[F; 2]>
    {
        let mut force = vec![[0.0, 0.0]; atoms.len()];

        let atoms: Vec<(usize, &Atom)> = atoms.iter()
            .enumerate()
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

    pub fn gravity_from_group<'a, I: Iterator<Item = &'a Atom>>(self, from: I, world_size: [F; 2], g: F, power: u8) -> [F; 2]
    {
        from
            .filter_map(|other| if self.pos != other.pos
            {
                Some(other.pos)
            }
            else
            {
                None
            })
            .map(|from| self.gravity_from(from, world_size, g, power))
            .reduce(|accum, gravity| [
                accum[0] + gravity[0],
                accum[1] + gravity[1]
            ])
            .unwrap_or([0.0, 0.0])
    }
}