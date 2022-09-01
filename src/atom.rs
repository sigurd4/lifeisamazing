pub mod species;

type F = f32;

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

    pub fn dist_to(&self, pos: [F; 2]) -> [F; 2]
    {
        [
            pos[0] - self.pos[0],
            pos[1] - self.pos[1]
        ]
    }

    fn gravity_from(&self, from: &Self, g: F, power: u8) -> [F; 2]
    {
        let d = self.dist_to(from.pos);
        let d_abs = d[0].hypot(d[1]);
        let f_abs = g/d_abs.powi(power as i32);

        [
            f_abs*d[0],
            f_abs*d[1]
        ]
    }

    pub fn gravity_all(atoms: &[Self], g: F, power: u8) -> Vec<[F; 2]>
    {
        let mut force = vec![[0.0, 0.0]; atoms.len()];

        atoms
            .iter()
            .enumerate()
            .for_each(|(i, atom)| (&atoms[0..i])
                .iter()
                .filter(|from| atom.pos != from.pos)
                .enumerate()
                .for_each(|(j, from)|
                {
                    let g = atom.gravity_from(&from, g, power);
                    
                    force[i][0] += g[0];
                    force[i][1] += g[1];

                    force[j][0] -= g[0];
                    force[j][1] -= g[1];
                })
            );
        
        force
    }

    pub fn gravity_from_group<'a, I: Iterator<Item = &'a Atom>>(self, from: I, g: F, power: u8) -> [F; 2]
    {
        from
            .filter(|other| self.pos != other.pos)
            .map(|b| self.gravity_from(b, g, power))
            .reduce(|accum, gravity| [
                accum[0] + gravity[0],
                accum[1] + gravity[1]
            ])
            .unwrap_or([0.0, 0.0])
    }
}