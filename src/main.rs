mod operators;
mod model_store;

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    nn::{self, Linear, RmsNormConfig},
    tensor::{activation::silu, backend::Backend, Float, Shape, Tensor},
};
use operators::{Apply, OuterProduct};

const CONTEXT_SIZE: usize = 512;

struct Config {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

impl Config {
    fn config_7b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 32,
            n_head: 32,
            n_embd: 4096,
        }
    }

    fn config_13b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 40,
            n_head: 40,
            n_embd: 5120,
        }
    }

    fn config_30b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 60,
            n_head: 52,
            n_embd: 6656,
        }
    }

    fn config_65b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 80,
            n_head: 64,
            n_embd: 8192,
        }
    }
}

fn main() {
    let device = WgpuDevice::DiscreteGpu(0);
}

fn precompute_freq_cis<B: Backend>(config: Config, device: &B::Device) -> Tensor<B, 5> {
    let seq_len = CONTEXT_SIZE;
    let n_elem = config.n_embd / config.n_head;
    let theta =
        Tensor::<B, 1, _>::arange_step(0..(n_elem as i64), 2, device).float() / n_elem as f32;
    let arange = Tensor::<B, 1, _>::arange(0..seq_len as i64, device).float();
    let idx_theta = theta.outer(&arange);
    let shape = Shape::new([1, 1, seq_len, n_elem / 2, 1]);
    let idx_theta_cos = idx_theta.clone().cos().reshape(shape.clone());
    let idx_theta_sin = idx_theta.sin().reshape(shape);
    Tensor::<B, 5>::cat([idx_theta_cos, idx_theta_sin].to_vec(), 4)
}

#[test]
fn test_precompute_freq_cis() {
    let device = WgpuDevice::DiscreteGpu(0);
    let freq_cis = precompute_freq_cis::<Wgpu>(Config::config_7b(), &device);
    println!("{}", freq_cis);
}

#[derive(Debug)]
struct Mlp<B: Backend> {
    c_fc1: nn::Linear<B>,
    c_fc2: nn::Linear<B>,
    c_proj: nn::Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let xs = silu(xs.apply(&self.c_fc1)) * xs.apply(&self.c_fc2);
        xs.apply(&self.c_proj)
    }
}
