import torch

from attention.self_attention import GroupedQueryAttention, MultiHeadAttention, CausalAttention

if __name__ == '__main__':
    # model = MultiHeadAttention(hidden_dim=128, output_dim=32, num_head=8)
    # model = CausalAttention(hidden_dim=128, output_dim=32, num_head=8)
    model = GroupedQueryAttention(hidden_dim=128, num_heads=4, group_size=2)
    input_x = torch.rand((8,77,128))
    output_x = model(input_x)
    print(output_x.shape)