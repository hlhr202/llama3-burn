use burn::tensor::{backend::Backend, bf16, f16, Bool, Data, Device, Float, Int, Shape, Tensor};
use safetensors::{Dtype, SafeTensors};

#[test]
fn test_read_tinyllama2() {
    use burn::backend::{
        candle::{Candle, CandleDevice},
        wgpu::{Wgpu, WgpuDevice},
    };
    use std::io::Read;

    let file = std::fs::File::open("model.safetensors").unwrap();
    let bytes = file.bytes().collect::<Result<Vec<u8>, _>>().unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();

    tensors.tensors().into_iter().for_each(|(name, view)| {
        let shape = view.shape();
        println!("{}: {:?}, {:?}", name, shape.to_vec(), view.dtype());
    });

    println!();

    #[cfg(not(target_os = "macos"))]
    let device = CandleDevice::Cpu;

    #[cfg(target_os = "macos")]
    let device = CandleDevice::Metal(0);

    let up_proj_bf16 = tensors
        .read_burn_tensor_bf16::<Candle<bf16>, 2>("model.layers.0.mlp.up_proj.weight", &device);
    println!("{}", up_proj_bf16);

    let device = WgpuDevice::DiscreteGpu(0);
    // wgpu is not supporting bf16
    let up_proj_f32 = tensors
        .read_burn_tensor_bf16_as_f32::<Wgpu, 2>("model.layers.0.mlp.up_proj.weight", &device);
    println!("{}", up_proj_f32);
}

#[test]
fn test_serialize() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
    use std::collections::HashMap;

    let raw = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data: Vec<u8> = raw
        .clone()
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let shape = vec![1, 1, 2, 3];
    let attn_0 = TensorView::new(Dtype::F32, shape.clone(), &data).unwrap();
    let metadata: HashMap<String, TensorView> =
            // Smaller string to force misalignment compared to previous test.
            [("attn0".to_string(), attn_0)].into_iter().collect();
    let binary = serialize(&metadata, &None).unwrap();

    // ====================
    let parsed = SafeTensors::deserialize(&binary).unwrap();
    let device = WgpuDevice::DiscreteGpu(0);
    let tensor = parsed.read_burn_tensor_f32::<Wgpu, 4>("attn0", &device);

    let result = tensor.to_data().value;

    assert_eq!(result, raw);
    assert_eq!(shape, tensor.shape().dims);
}

trait ReadBurnTensor {
    fn read_burn_tensor_bool<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Bool>
    where
        Data<bool, D>: From<Data<bool, D>>;

    fn read_burn_tensor_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<f32, D>>;

    fn read_burn_tensor_f16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<f16, D>>;

    fn read_burn_tensor_bf16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<bf16, D>>;

    fn read_burn_tensor_bf16_as_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<f32, D>>;

    fn read_burn_tensor_i32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Int>
    where
        Data<B::IntElem, D>: From<Data<i32, D>>;
}

impl ReadBurnTensor for SafeTensors<'_> {
    fn read_burn_tensor_bool<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Bool>
    where
        Data<bool, D>: From<Data<bool, D>>,
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::BOOL);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[bool] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Data<bool, D> = Data::new(data.to_vec(), Shape::from(shape.to_vec()));
        Tensor::<B, D, Bool>::from_data(data, device)
    }

    fn read_burn_tensor_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<f32, D>>,
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::F32);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[f32] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Data<f32, D> = Data::new(data.to_vec(), Shape::from(shape.to_vec()));
        Tensor::<B, D, Float>::from_data(data, device)
    }

    fn read_burn_tensor_f16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<f16, D>>,
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::F16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };

        let data: Vec<f16> = data
            .iter()
            .map(|&x| f16::from_bits(x))
            .collect::<Vec<f16>>();
        let data: Data<f16, D> = Data::new(data, Shape::from(shape.to_vec()));
        Tensor::<B, D, Float>::from_data(data, device)
    }

    fn read_burn_tensor_bf16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<bf16, D>>,
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::BF16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Vec<bf16> = data
            .iter()
            .map(|&x| bf16::from_bits(x))
            .collect::<Vec<bf16>>();
        let data: Data<bf16, D> = Data::new(data, Shape::from(shape.to_vec()));
        Tensor::<B, D, Float>::from_data(data, device)
    }

    fn read_burn_tensor_bf16_as_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    where
        Data<B::FloatElem, D>: From<Data<f32, D>>,
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::BF16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Vec<f32> = data
            .iter()
            .map(|&x| bf16::from_bits(x).to_f32())
            .collect::<Vec<f32>>();
        let data: Data<f32, D> = Data::new(data, Shape::from(shape.to_vec()));
        Tensor::<B, D, Float>::from_data(data, device)
    }

    fn read_burn_tensor_i32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Int>
    where
        Data<B::IntElem, D>: From<Data<i32, D>>,
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::I32);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[i32] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Data<i32, D> = Data::new(data.to_vec(), Shape::from(shape.to_vec()));
        Tensor::<B, D, Int>::from_data(data, device)
    }
}
