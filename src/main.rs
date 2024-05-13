mod config;
mod loader;
mod model;
mod operators;

use std::io::Read;

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    module::Param,
    nn::{self, Embedding},
    tensor::{backend::Backend, Data, Int, Shape, Tensor},
};
use config::Config;
use loader::SafeTensorsReader;
use model::{Mlp, MultiHeadSelfAttention, ResidualDecoderAttentionBlock, RmsNorm};
use num_traits::ToPrimitive;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use crate::model::{attn_decoder_mask, Llama, RotaryEncodingConfig};

fn load_mlp<B: Backend>(safetensors: &SafeTensors, path: &str, device: &B::Device) -> Mlp<B>
where
    Data<B::FloatElem, 2>: From<Data<f32, 2>>,
{
    let proj = safetensors.tensor(&format!("{}.gate_proj.weight", path)).unwrap();
    println!("{:?}", proj.shape());
    let gate_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.gate_proj.weight", path), device);
    let up_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.up_proj.weight", path), device);
    let down_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.down_proj.weight", path), device);
    Mlp {
        gate_proj: nn::Linear {
            weight: Param::from_tensor(gate_proj),
            bias: None,
        },
        up_proj: nn::Linear {
            weight: Param::from_tensor(up_proj),
            bias: None,
        },
        down_proj: nn::Linear {
            weight: Param::from_tensor(down_proj),
            bias: None,
        },
    }
}

fn load_rmsnorm<B: Backend>(
    safetensors: &SafeTensors,
    config: &Config,
    path: &str,
    device: &B::Device,
) -> RmsNorm<B>
where
    Data<B::FloatElem, 1>: From<Data<f32, 1>>,
{
    let weight =
        safetensors.read_burn_tensor_bf16_as_f32::<B, 1>(&format!("{}.weight", path), device);
    let eps = config.rms_norm_eps;
    RmsNorm {
        weight: Param::from_tensor(weight),
        eps,
    }
}

fn load_attention<B: Backend>(
    safetensors: &SafeTensors,
    config: &Config,
    path: &str,
    device: &B::Device,
) -> MultiHeadSelfAttention<B>
where
    Data<B::FloatElem, 2>: From<Data<f32, 2>>,
{
    let k_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.k_proj.weight", path), device);
    let q_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.q_proj.weight", path), device);
    let v_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.v_proj.weight", path), device);
    let o_proj = safetensors
        .read_burn_tensor_bf16_as_f32::<B, 2>(&format!("{}.o_proj.weight", path), device);

    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    MultiHeadSelfAttention {
        n_heads,
        n_kv_heads,
        query: nn::Linear {
            weight: Param::from_tensor(q_proj),
            bias: None,
        },
        key: nn::Linear {
            weight: Param::from_tensor(k_proj),
            bias: None,
        },
        value: nn::Linear {
            weight: Param::from_tensor(v_proj),
            bias: None,
        },
        out: nn::Linear {
            weight: Param::from_tensor(o_proj),
            bias: None,
        },
    }
}

fn load_transformer_block<B: Backend>(
    safetensors: &SafeTensors,
    config: &Config,
    path: &str,
    device: &B::Device,
) -> ResidualDecoderAttentionBlock<B>
where
    Data<B::FloatElem, 2>: From<Data<f32, 2>>,
    Data<B::FloatElem, 1>: From<Data<f32, 1>>,
{
    let self_attn =
        load_attention::<B>(safetensors, config, &format!("{}.self_attn", path), device);
    let input_layernorm = load_rmsnorm::<B>(
        safetensors,
        config,
        &format!("{}.input_layernorm", path),
        device,
    );
    let mlp = load_mlp::<B>(safetensors, &format!("{}.mlp", path), device);
    let post_attn_layernorm = load_rmsnorm::<B>(
        safetensors,
        config,
        &format!("{}.post_attention_layernorm", path),
        device,
    );

    ResidualDecoderAttentionBlock {
        self_attn,
        input_layernorm,
        mlp,
        post_attn_layernorm,
    }
}

fn load_llama<B: Backend>(
    safetensors: &SafeTensors,
    config: &Config,
    device: &B::Device,
) -> Llama<B>
where
    Data<B::FloatElem, 2>: From<Data<f32, 2>>,
    Data<B::FloatElem, 1>: From<Data<f32, 1>>,
{
    let mut blocks: Vec<ResidualDecoderAttentionBlock<B>> = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let transformer_block = load_transformer_block::<B>(
            safetensors,
            config,
            &format!("model.layers.{}", i),
            device,
        );
        blocks.push(transformer_block);
    }

    let embed_tokens =
        safetensors.read_burn_tensor_bf16_as_f32::<B, 2>("model.embed_tokens.weight", device);

    let [_n_vacab, n_state] = embed_tokens.dims();
    let n_heads = config.n_heads;
    let _n_kv_heads = config.n_kv_heads;
    let head_dim = n_state / n_heads;
    let token_embedding = Embedding {
        weight: Param::from_tensor(embed_tokens),
    };
    let rotary_encoding =
        RotaryEncodingConfig::new(config.max_seq_len, head_dim, config.rope_theta).init(device);
    let norm = load_rmsnorm::<B>(safetensors, config, "model.norm", device);
    // sometimes lm_head is also called "output"
    let lm_head = safetensors.read_burn_tensor_bf16_as_f32::<B, 2>("lm_head.weight", device);
    let lm_head = nn::Linear {
        weight: Param::from_tensor(lm_head),
        bias: None,
    };
    let mask = attn_decoder_mask::<B>(config.max_seq_len, device);
    let _norm_eps = norm.eps;

    Llama {
        token_embedding,
        rotary_encoding,
        blocks,
        norm,
        lm_head,
        mask,
        max_seq_len: config.max_seq_len,
    }
}

fn main() {
    let device = WgpuDevice::default();

    let path = "model.safetensors";
    let file = std::fs::File::open(path).unwrap();
    let bytes = file.bytes().collect::<Result<Vec<u8>, _>>().unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let config = Config::config_tiny();

    let llama = load_llama::<Wgpu>(&tensors, &config, &device);

    println!("{:#?}", llama);
    let tokenizer = Tokenizer::from_pretrained("stas/tiny-random-llama-2", None).unwrap();
    let prompt = "Hello, ".to_string();
    let tokens = tokenizer.encode(prompt, false).unwrap();
    let mut tokens = tokens
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();

    let mut text = String::new();

    for _i in 0..256 {
        let input = Data::new(tokens.clone(), Shape::new([tokens.len()]));
        let input = Tensor::<Wgpu, 1, Int>::from_data(input, &device).unsqueeze();

        let out = llama.forward(input);
        let [_n_batch, n_token, _n_dict] = out.dims();
        let last_row: Tensor<Wgpu, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

        let token_id = last_row.argmax(0).into_scalar().to_i32().unwrap();
        tokens.push(token_id);

        let token_text = tokenizer.decode(&[token_id as u32], true).unwrap();
        println!("{token_text}");

        text += &token_text;
    }
}
