pub struct Species
{
    pub id: u8,
    pub size: f64,
    pub color: [f32; 4]
}

impl Species
{
    pub fn index(&self) -> usize
    {
        self.id as usize
    }
}