use crate::operators::{OuterProduct, Rsqrt};
use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{
        activation::{silu, softmax},
        backend::Backend,
        Int, Shape, Tensor,
    },
};
use std::f32::NEG_INFINITY;

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rotary_encoder: &RotaryEncoding<B>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let wv = qkv_attention_rotary(q, k, v, mask, self.n_heads, self.n_kv_heads, rotary_encoder);

        self.out.forward(wv)
    }
}

fn qkv_attention_rotary<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
    n_kv_head: usize,
    rotary_encoder: &RotaryEncoding<B>,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let n_hstate = n_state / n_head;
    let scale = (n_hstate as f64).powf(-0.25); // keeps the value weightings roughly normally distributed

    let q = q.reshape([n_batch, n_qctx, n_head, n_hstate]);
    // interleave kv heads to match the number of q heads
    let n_repeat = n_head / n_kv_head;
    let k = repeat_kv(k.reshape([n_batch, n_ctx, n_kv_head, n_hstate]), n_repeat);
    let v = repeat_kv(v.reshape([n_batch, n_ctx, n_kv_head, n_hstate]), n_repeat);

    // the last two dims need to be (n_ctx, n_hstate)
    let q = rotary_encoder.forward(q.swap_dims(1, 2)) * scale;
    let k = rotary_encoder.forward(k.swap_dims(1, 2)) * scale;
    let v = v.swap_dims(1, 2);

    // compute value weightings
    let qk = q.matmul(k.transpose());

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    // normalize value weightings
    let w = softmax(qk, 3);

    // output
    w.matmul(v).swap_dims(1, 2).flatten(2, 3)
}

/// For a tensor of size (n_batch, n_ctx, n_kv_head, n_hstate), repeats the head keys or values in an interleaving manner so that the number
/// of heads is effectively multiplied by n_repeat
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_repeat: usize) -> Tensor<B, 4> {
    if n_repeat > 1 {
        let [n_batch, n_ctx, n_kv_head, n_hstate] = x.dims();
        x.repeat(3, n_repeat)
            .reshape([n_batch, n_ctx, n_kv_head * n_repeat, n_hstate])
    } else {
        x
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + self.eps).rsqrt();
        let x = x * rms;

        x * self.weight.val().unsqueeze()
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    // w1
    pub gate_proj: nn::Linear<B>,
    // w3
    pub up_proj: nn::Linear<B>,
    // w2
    pub down_proj: nn::Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let xs = silu(self.gate_proj.forward(xs.clone())) * self.up_proj.forward(xs.clone());
        self.down_proj.forward(xs)
    }
}

#[derive(Module, Debug)]
pub struct RotaryEncoding<B: Backend> {
    arange_m: Tensor<B, 2>,
    freq_cis: Tensor<B, 3>,
}

impl<B: Backend> RotaryEncoding<B> {
    /// Applies rotary positional encoding to a tensor of dimenions (..., seq_len, n_state)
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        assert!(D >= 2);
        let orig_shape = x.shape();
        let (n_ctx, n_state) = (orig_shape.dims[D - 2], orig_shape.dims[D - 1]);
        let dummy_dim_size = orig_shape.num_elements() / (n_ctx * n_state);

        #[allow(clippy::single_range_in_vec_init)]
        let out = x
            .reshape([dummy_dim_size, n_ctx, n_state / 2, 2])
            .matmul(self.arange_m.clone().unsqueeze())
            .reshape([dummy_dim_size, n_ctx, n_state, 2])
            * self.freq_cis.clone().slice([0..n_ctx]).unsqueeze();

        out.sum_dim(D - 1).reshape(orig_shape)
    }
}

#[derive(Config, Debug)]
pub struct RotaryEncodingConfig {
    pub max_sequence_length: usize,
    pub state_size: usize,
    pub theta: f64,
}

impl RotaryEncodingConfig {
    // not in use
    pub fn _precompute_freq_cis<B: Backend>(
        config: crate::Config,
        device: &B::Device,
    ) -> Tensor<B, 5> {
        let seq_len = config.max_seq_len;
        let n_elem = config.n_embd / config.n_heads;
        let theta =
            Tensor::<B, 1, _>::arange_step(0..(n_elem as i64), 2, device).float() / n_elem as f32;
        let arange = Tensor::<B, 1, _>::arange(0..seq_len as i64, device).float();
        let idx_theta = theta.outer(&arange);
        let shape = Shape::new([1, 1, seq_len, n_elem / 2, 1]);
        let idx_theta_cos = idx_theta.clone().cos().reshape(shape.clone());
        let idx_theta_sin = idx_theta.sin().reshape(shape);
        Tensor::<B, 5>::cat([idx_theta_cos, idx_theta_sin].to_vec(), 4)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> RotaryEncoding<B> {
        assert!(self.state_size % 2 == 0, "Head dims must be even.");
        assert!(self.theta > 0.0, "Theta must be positive.");

        let half_state_size = self.state_size / 2;

        let arange_m = Tensor::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], device);

        let inv_freq = powto(
            self.theta,
            Tensor::arange(0..half_state_size as i64, device).float()
                * (2.0 / self.state_size as f64),
        )
        .powf_scalar(-1.0);

        let periods = Tensor::arange(0..self.max_sequence_length as i64, device)
            .float()
            .unsqueeze::<2>()
            .transpose()
            .repeat(1, half_state_size)
            * inv_freq.unsqueeze();

        let p_cos = periods.clone().cos();
        let p_sin = periods.sin();
        let freq_cis = Tensor::cat(vec![p_cos, p_sin], 1)
            .reshape([self.max_sequence_length, 2, half_state_size])
            .transpose()
            .repeat(2, 2)
            .reshape([self.max_sequence_length, self.state_size, 2]);

        RotaryEncoding { arange_m, freq_cis }
    }
}

fn powto<B: Backend, const D: usize>(base: f64, x: Tensor<B, D>) -> Tensor<B, D> {
    let logbase = base.ln();
    x.mul_scalar(logbase).exp()
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub self_attn: MultiHeadSelfAttention<B>,
    pub input_layernorm: RmsNorm<B>,
    pub mlp: Mlp<B>,
    pub post_attn_layernorm: RmsNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        rotary_encoder: &RotaryEncoding<B>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self
            .self_attn
            .forward(hidden_states, rotary_encoder, Some(mask));
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attn_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }
}

pub fn attn_decoder_mask<B: Backend>(seq_length: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::full([seq_length, seq_length], NEG_INFINITY, device).triu(1)
}

#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    pub token_embedding: nn::Embedding<B>,
    pub rotary_encoding: RotaryEncoding<B>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: nn::Linear<B>,
    pub mask: Tensor<B, 2>,
    // pub n_vocab: usize,
    pub max_seq_len: usize,
}

impl<B: Backend> Llama<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.max_seq_len,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_seq_len
        );

        let x = self.token_embedding.forward(x);

        let x = self.blocks.iter().fold(x, |acc, block| {
            block.forward(acc, &self.rotary_encoding, self.mask.clone())
        });

        self.lm_head.forward(self.norm.forward(x))
    }
}
