pub struct Config {
    pub block_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden_size: usize,
    pub n_embd: usize,
    pub rms_norm_eps: f64,
    // n_ctx
    pub max_seq_len: usize,
    pub rope_theta: f64,
}

impl Config {
    pub fn config_tiny() -> Self {
        Self {
            block_size: 512,
            vocab_size: 256,
            n_layer: 2,
            n_heads: 4,
            n_kv_heads: 4,
            hidden_size: 16,
            n_embd: 256,
            rms_norm_eps: 1e-05,
            // https://github.com/aju22/LLaMA2/blob/5716de40720123bf03013f3e08673a7e0feb53ba/model.py#L19
            max_seq_len: 2048,
            rope_theta: 10000.0,
        }
    }
}
