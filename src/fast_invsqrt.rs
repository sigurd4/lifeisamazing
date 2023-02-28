use std::mem::transmute;

pub trait FastInvSqrt
{
    fn fast_invsqrt(self) -> Self;
}

impl FastInvSqrt for f32
{
    fn fast_invsqrt(self) -> Self
    {
        let x_half = self*0.5;

        let i = unsafe {
            transmute::<_, u32>(self)
        };
        let i = 0x5f375a86 - (i >> 1);
        let y = unsafe {
            transmute::<_, f32>(i)
        };

        y*(1.5 - x_half*y*y)
    }
}

impl FastInvSqrt for f64
{
    fn fast_invsqrt(self) -> Self
    {
        let x_half = self*0.5;

        let i = unsafe {
            transmute::<_, u64>(self)
        };
        let i = 0x5fe6ec85e7de30da - (i >> 1);
        let y = unsafe {
            transmute::<_, f64>(i)
        };

        y*(1.5 - x_half*y*y)
    }
}