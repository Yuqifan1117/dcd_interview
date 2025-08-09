import torch
import torch.nn as nn
import torch.nn.functional as F


# multi-head self attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        k = self.q(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        v = self.q(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)

        # attention score
        attention_score = torch.matmul(q, k.transpose(-2,-1)) // self.head_dim ** 0.5
        # dont need mask
        attention_score = F.softmax(attention_score, dim=-1)

        attention_output = torch.matmul(attention_score, v)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.o(attention_output)


# causal attention
class CausalAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_head):
        super(CausalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        k = self.q(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        v = self.q(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)

        # attention score
        attention_score = torch.matmul(q, k.transpose(-2,-1)) // self.head_dim ** 0.5
        # causal mask
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        attention_score = attention_score.masked_fill(attention_mask==0, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)

        attention_output = torch.matmul(attention_score, v)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.o(attention_output)

# group query attention 
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_size):
        super(GroupedQueryAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = hidden_dim // num_heads
        self.group_num = num_heads // group_size

        # 线性矩阵得到 qkv

        self.q = nn.Linear(hidden_dim, hidden_dim)  # 分组共享
        self.k = nn.Linear(hidden_dim, self.group_num*self.head_dim)
        self.v = nn.Linear(hidden_dim, self.group_num*self.head_dim)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k(x).view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1,2)
        v = self.v(x).view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1,2)

        # repeat expand
        k = k.unsqueeze(2).expand(-1,-1,self.group_size,-1,-1)
        k = k.contiguous().view(batch_size, -1, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1,-1,self.group_size,-1,-1)
        v = v.contiguous().view(batch_size, -1, seq_len, self.head_dim)

        # attention weight
        attention_score = torch.matmul(q, k.transpose(-2,-1)) // self.head_dim**0.5
        if attention_mask is not None:
            attention_score = attention_score.masked_fill(attention_mask[:,None,None, :]==0, float('-inf'))
        
        attention_output = torch.matmul(attention_score, v)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return attention_output

