pub struct Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    // n_ctx or max_seq_len or max_position_embeddings
    // https://github.com/aju22/LLaMA2/blob/5716de40720123bf03013f3e08673a7e0feb53ba/model.py#L19
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
}

#[allow(dead_code)]
impl Config {
    // https://huggingface.co/stas/tiny-random-llama-2/blob/main/config.json
    pub fn tiny_llama2() -> Self {
        Self {
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-05,
            max_position_embeddings: 256,
            rope_theta: 10000.0,
        }
    }

    // https://huggingface.co/HuggingFaceM4/tiny-random-Llama3ForCausalLM/blob/main/config.json
    pub fn tiny_llama3() -> Self {
        Self {
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-06,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
        }
    }
}
