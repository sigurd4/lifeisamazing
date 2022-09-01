use crate::atom::species::Species;

type F = f32;

#[derive(Clone, Copy)]
pub enum Rule
{
    Gravity(&'static Species, F, u8),
    GravitySelf(F, u8)
}