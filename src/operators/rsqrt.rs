use burn::tensor::{backend::Backend, Tensor};

pub trait Rsqrt<B, const D: usize>
where
    B: Backend,
{
    fn rsqrt(&self) -> Tensor<B, D>;
}

impl<B, const D: usize> Rsqrt<B, D> for Tensor<B, D>
where
    B: Backend,
{
    fn rsqrt(&self) -> Tensor<B, D> {
        Tensor::ones(self.shape(), &self.device()) / self.clone().sqrt()
    }
}

#[test]
fn test_rsqrt() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    let device = WgpuDevice::BestAvailable;
    let a = Tensor::<Wgpu, 1>::from_data([1., 2., 3., 4.], &device);
    let result = a.rsqrt();
    let expected = Tensor::ones([4], &device) / a.sqrt();

    assert_eq!(result.to_data(), expected.to_data());
}
