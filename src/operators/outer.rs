use burn::tensor::{backend::Backend, Numeric, Shape, Tensor};

pub trait OuterProduct<B, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: burn::tensor::Element,
{
    fn outer(&self, other: &Self) -> Tensor<B, 2, K>;
}

impl<B, K> OuterProduct<B, K> for Tensor<B, 1, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: burn::tensor::Element,
{
    fn outer(&self, other: &Self) -> Tensor<B, 2, K> {
        let other = other
            .clone()
            .reshape(Shape::new([1, other.shape().dims[0]]));
        self.clone()
            .reshape(Shape::new([self.shape().dims[0], 1]))
            .mul(other)
    }
}

#[test]
fn test_outer() {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    let device = NdArrayDevice::Cpu;
    let v1 = Tensor::<NdArray, 1, _>::arange(1..5, &device).float();
    let v2 = Tensor::<NdArray, 1, _>::arange(1..4, &device).float();
    let result = v1.outer(&v2);
    let expected = Tensor::<NdArray, 2, _>::from_floats(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ],
        &device,
    );

    println!("result: {}", result);

    assert_eq!(result.to_data(), expected.to_data());
}
