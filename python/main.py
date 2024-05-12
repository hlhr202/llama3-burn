import torch

torch.set_printoptions(precision=6, sci_mode=False)

context_size = 512
block_size = 4096
vocab_size = 32000
n_layer = 32
n_head = 32
n_embd = 4096


def precompute_freq_cis():
    seq_len = context_size
    n_elem = int(n_embd / n_head)
    theta = torch.arange(0, n_elem, 2).float() / float(n_elem)
    arange = torch.arange(0, seq_len).float()
    idx_theta = theta.outer(arange)
    shape = [1, 1, seq_len, n_elem // 2, 1]
    idx_theta_cos = idx_theta.cos().reshape(shape)
    idx_theta_sin = idx_theta.sin().reshape(shape)
    return torch.cat([idx_theta_cos, idx_theta_sin], dim=-1)


freq_cis = precompute_freq_cis()

print(freq_cis)
print(freq_cis.shape)
