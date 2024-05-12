use burn::{
    nn,
    tensor::{backend::Backend, Tensor},
};

pub trait Apply<B, const D: usize>
where
    B: Backend,
{
    fn apply(&self, other: &nn::Linear<B>) -> Tensor<B, D>;
}

impl<B, const D: usize> Apply<B, D> for Tensor<B, D>
where
    B: Backend,
{
    fn apply(&self, other: &nn::Linear<B>) -> Tensor<B, D> {
        other.forward(self.clone())
    }
}
