# type: ignore
# pylint: disable=undefined-variable

from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from einops import rearrange, repeat  # For self-documenting tensor transformations


class Embedding(nn.Module):
    """
    An Embedding module that performs embedding table lookup for token IDs.
    
    This class implements an embedding layer that converts discrete token IDs into
    dense vector representations. Each token ID is mapped to a learned embedding vector.
    
    The embedding table is a learnable parameter matrix where each row represents 
    the embedding vector for a specific token in the vocabulary.
    
    Args:
        num_embeddings (int): Size of the vocabulary (number of possible token IDs)
        embedding_dim (int): Dimension of the embedding vectors (d_model)
        device (torch.device | None): Device to place the parameters on (CPU/GPU)
        dtype (torch.dtype | None): Data type for the parameters (float32, float64, etc.)
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # Call the parent class constructor to properly initialize the nn.Module
        # This sets up important internal state for parameter tracking and gradient computation
        super().__init__()

        # Store the dimensions for reference and validation
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create the embedding weight matrix with shape (num_embeddings, embedding_dim)
        # Each row represents the embedding vector for a specific token ID
        # The final dimension is embedding_dim (d_model) as specified in instructions
        # nn.Parameter wraps a tensor and tells PyTorch this is a learnable parameter
        # that should be included in gradients and optimizer updates
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # Initialize the embedding weights using truncated normal distribution
        # This initialization helps with stable training and prevents vanishing/exploding gradients
        # trunc_normal_ initializes values from a truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embedding vectors for the given token IDs.
        
        This method performs an embedding table lookup operation. For each token ID
        in the input tensor, it retrieves the corresponding embedding vector from
        the weight matrix.
        
        The operation is: output[..., i, :] = weight[token_ids[..., i], :]
        where token_ids[..., i] is used as an index into the embedding table.
        
        Args:
            token_ids (torch.Tensor): Input tensor of token IDs with shape (...,)
                Each value should be an integer in range [0, num_embeddings-1]
                The ... represents arbitrary leading dimensions (batch size, sequence length, etc.)
                
        Returns:
            torch.Tensor: Output tensor of embeddings with shape (..., embedding_dim)
                Same leading dimensions as input, with embedding_dim as the final dimension
        """
        # Perform embedding lookup using advanced indexing
        # token_ids contains indices into the embedding table
        # self.weight[token_ids] retrieves the embedding vectors for each token ID
        # 
        # For example:
        # - token_ids shape: (batch_size, seq_len)  
        # - weight shape: (num_embeddings, embedding_dim)
        # - output shape: (batch_size, seq_len, embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) module.
    
    RMSNorm is a normalization technique that normalizes the input by the root mean square (RMS)
    of the input elements. Unlike LayerNorm, RMSNorm does not subtract the mean, which makes it
    simpler and often more stable for training large language models.
    
    The mathematical operation is:
    output = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        d_model (int): Hidden dimension of the model (size of the last dimension)
        eps (float): Small value added to denominator for numerical stability (default: 1e-5)
        device (torch.device | None): Device to place the parameters on (CPU/GPU)
        dtype (torch.dtype | None): Data type for the parameters (float32, float64, etc.)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # Call the parent class constructor to properly initialize the nn.Module
        # This sets up important internal state for parameter tracking and gradient computation
        super().__init__()
        
        # Store the model dimension and epsilon for numerical stability
        self.d_model = d_model
        self.eps = eps
        
        # Create the learnable scale parameter (weight)
        # This is a vector of size d_model that scales each feature dimension
        # nn.Parameter wraps a tensor and tells PyTorch this is a learnable parameter
        # that should be included in gradients and optimizer updates
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        
        # Initialize the weights using truncated normal distribution
        # This initialization helps with stable training and prevents vanishing/exploding gradients
        # trunc_normal_ initializes values from a truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        This method normalizes the input by dividing by the RMS (root mean square) of the
        input elements along the last dimension, then scales by learnable weights.
        
        The normalization is performed in float32 for numerical stability, then converted
        back to the original dtype.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)
                The last dimension must match self.d_model
                Can have arbitrary leading dimensions (batch_size, sequence_length, etc.)
                
        Returns:
            torch.Tensor: Output tensor of same shape as input (..., d_model)
                The normalized and scaled tensor
        """
        # Store the original dtype to convert back later
        input_dtype = x.dtype
        
        # Upcast to float32 for numerical stability during normalization
        # This prevents precision issues that can occur with float16/bfloat16
        x = x.to(torch.float32)
        
        # Compute the root mean square (RMS) along the last dimension
        # Step 1: Square all elements: x^2
        x_squared = x * x
        
        # Step 2: Compute mean along the last dimension (d_model), keeping dimensions
        # keepdim=True preserves the dimension for broadcasting
        mean_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        
        # Step 3: Add epsilon for numerical stability and take square root to get RMS
        # eps prevents division by zero when all elements are zero
        rms = torch.sqrt(mean_squared + self.eps)
        
        # Normalize by dividing by RMS
        # This makes the RMS of the normalized vector equal to 1
        normalized = x / rms
        
        # Scale by learnable weight parameters
        # The weight vector is broadcast across all dimensions except the last
        # normalized shape: (..., d_model)
        # self.weight shape: (d_model,)
        # Result shape: (..., d_model)
        output = normalized * self.weight
        
        # Convert back to original dtype
        # This is important for mixed precision training and memory efficiency
        output = output.to(input_dtype)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network.
    
    SwiGLU is a variant of GLU (Gated Linear Unit) that uses SiLU (Swish) activation.
    It's commonly used in transformer models as the position-wise feed-forward network.
    
    The architecture consists of three linear transformations:
    - W1: projects input from d_model to d_ff (gate projection)
    - W2: projects from d_ff back to d_model (output projection)  
    - W3: projects input from d_model to d_ff (value projection)
    
    The computation is: W2(SiLU(W1(x)) ⊙ W3(x))
    where ⊙ represents element-wise multiplication (gating mechanism)
    
    Args:
        d_model (int): Input and output dimension (hidden size)
        d_ff (int): Inner feed-forward dimension (typically ~8/3 * d_model)
        device (torch.device | None): Device to place the parameters on
        dtype (torch.dtype | None): Data type for the parameters
    """
    
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        # Call the parent class constructor to properly initialize the nn.Module
        # This sets up important internal state for parameter tracking and gradient computation
        super().__init__()
        
        # Store dimensions for reference
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Create the three linear transformations used in SwiGLU
        # W1: Gate projection - transforms input to intermediate dimension with SiLU activation
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        
        # W2: Output projection - transforms intermediate dimension back to output dimension
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        
        # W3: Value projection - transforms input to intermediate dimension (no activation)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward transformation to the input.
        
        The computation follows this pattern:
        1. Gate path: W1(x) -> SiLU activation
        2. Value path: W3(x) -> no activation  
        3. Element-wise multiplication (gating): SiLU(W1(x)) ⊙ W3(x)
        4. Output projection: W2(gated_result)
        
        This gating mechanism allows the network to selectively pass information,
        where the SiLU-activated gate controls what information from the value path
        gets passed through.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)
                Can have arbitrary leading dimensions (batch_size, sequence_length, etc.)
                
        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
                Same leading dimensions as input, with d_model as final dimension
        """
        # Gate path: Apply W1 linear transformation and SiLU activation
        # W1 projects from d_model to d_ff, then SiLU provides non-linearity
        gate = run_silu(self.w1(x))  # Shape: (..., d_ff)
        
        # Value path: Apply W3 linear transformation (no activation)
        # W3 also projects from d_model to d_ff
        value = self.w3(x)  # Shape: (..., d_ff)
        
        # Gating mechanism: Element-wise multiplication
        # The SiLU-activated gate controls which information from value gets passed
        # This is the key innovation of GLU - selective information flow
        gated = gate * value  # Shape: (..., d_ff)
        
        # Output projection: Apply W2 to project back to original dimension
        # W2 projects from d_ff back to d_model
        output = self.w2(gated)  # Shape: (..., d_model)
        
        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.
    
    RoPE applies pairwise rotation matrices to query and key vectors based on their position.
    For a query token q(i) at position i, we apply rotation matrix Ri giving q'(i) = Ri * q(i).
    
    The rotation matrix Ri is block-diagonal with 2x2 rotation blocks:
    - Each block Ri_k rotates embedding elements q(i)[2k-1:2k] by angle θi,k = i / Θ^((2k-1)/d)
    - Ri_k = [[cos(θi,k), -sin(θi,k)], [sin(θi,k), cos(θi,k)]]
    - Full matrix Ri has blocks Ri_1, Ri_2, ..., Ri_(d/2) on the diagonal
    
    This implementation efficiently applies the rotation without constructing the full d×d matrix.
    
    Args:
        theta (float): Base frequency parameter Θ (typically 10000)
        d_k (int): Dimension of query/key vectors (must be even)
        max_seq_len (int): Maximum sequence length for precomputing cos/sin values
        device (torch.device | None): Device to store the precomputed buffers
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # Call the parent class constructor to properly initialize the nn.Module
        super().__init__()
        
        # Store parameters for reference
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Ensure d_k is even since RoPE works on pairs of dimensions
        assert d_k % 2 == 0, f"d_k must be even for RoPE, got {d_k}"
        
        # Precompute rotation frequencies for each dimension pair k ∈ {1, ..., d/2}
        # For pair k, we use dimensions [2k-1, 2k] (0-indexed: [2k-2, 2k-1])
        # The frequency is: 1 / Θ^((2k-1)/d) = 1 / Θ^((2*(k-1))/d) for k ∈ {1, ..., d/2}
        # In 0-indexed terms: 1 / Θ^(2*k/d) for k ∈ {0, 1, ..., d/2-1}
        k = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (2.0 * k / d_k))  # Shape: (d_k/2,)
        
        # Precompute positions for the maximum sequence length
        # positions = [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # Shape: (max_seq_len,)
        
        # Compute the rotation angles for all position-frequency combinations
        # θi,k = position_i * frequency_k
        angles = torch.outer(positions, freqs)  # Shape: (max_seq_len, d_k/2)
        
        # Precompute cosine and sine values for all angles
        # These will be used in the 2x2 rotation matrices
        cos_cached = torch.cos(angles)  # Shape: (max_seq_len, d_k/2)
        sin_cached = torch.sin(angles)  # Shape: (max_seq_len, d_k/2)
        
        # Register as non-persistent buffers (not saved in state_dict but moved with module)
        # These are precomputed values, not learnable parameters
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to the input tensor efficiently.
        
        Instead of constructing the full block-diagonal matrix Ri, we directly apply
        the pairwise rotations to consecutive dimension pairs.
        
        For each pair of dimensions [2k-1, 2k] (0-indexed: [2k, 2k+1]):
        - Extract x_even = x[..., 2k] and x_odd = x[..., 2k+1] 
        - Apply rotation: 
          rotated_even = x_even * cos(θi,k) - x_odd * sin(θi,k)
          rotated_odd = x_even * sin(θi,k) + x_odd * cos(θi,k)
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k)
                Query or key vectors to apply RoPE to
            token_positions (torch.Tensor): Token positions of shape (..., seq_len)
                Position indices for each token in the sequence
                
        Returns:
            torch.Tensor: Rotated tensor of same shape (..., seq_len, d_k)
        """
        # Extract cos and sin values for the given token positions
        # token_positions shape: (..., seq_len)
        # cos/sin shape after indexing: (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]  # Shape: (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # Shape: (..., seq_len, d_k/2)
        
        # Reshape x to separate consecutive pairs for efficient rotation using einops
        # Transform: (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2) where last dim has [even, odd] elements
        # This groups consecutive dimensions into pairs for 2D rotation
        x_reshaped = rearrange(x, '... seq (pairs two) -> ... seq pairs two', two=2)
        
        # Extract even and odd indexed elements from each pair
        # These correspond to x[..., 0::2] and x[..., 1::2] respectively
        x_even = x_reshaped[..., 0]  # Shape: (..., seq_len, d_k/2) - dims 0, 2, 4, ...
        x_odd = x_reshaped[..., 1]   # Shape: (..., seq_len, d_k/2) - dims 1, 3, 5, ...
        
        # Apply the 2D rotation to each pair of dimensions
        # Rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        # Matrix multiplication gives:
        # [rotated_even] = [cos(θ)  -sin(θ)] [x_even]
        # [rotated_odd ]   [sin(θ)   cos(θ)] [x_odd ]
        rotated_even = x_even * cos - x_odd * sin  # Shape: (..., seq_len, d_k/2)
        rotated_odd = x_even * sin + x_odd * cos   # Shape: (..., seq_len, d_k/2)
        
        # Combine the rotated pairs back into the original format using einops
        # Stack the even and odd components back together
        rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1)  # Shape: (..., seq_len, d_k/2, 2)
        
        # Reshape back to the original tensor shape using einops
        # Transform: (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)
        # This interleaves the rotated even and odd elements correctly
        rotated_x = rearrange(rotated_pairs, '... seq pairs two -> ... seq (pairs two)')
        
        return rotated_x


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention as described in Vaswani et al. [2017].
    
    Computes: Attention(Q, K, V) = softmax(Q⊤K / √d_k) V
    
    This is the core attention mechanism used in transformers. It allows each query
    to attend to all keys, with attention weights determined by the similarity
    between queries and keys (scaled by √d_k for numerical stability).
    
    Mathematical steps:
    1. Compute attention scores: Q @ K⊤ 
    2. Scale by √d_k: scores / √d_k
    3. Apply mask (if provided): set False positions to -∞
    4. Apply softmax: convert scores to attention weights  
    5. Apply attention: weights @ V
    
    Args:
        Q (torch.Tensor): Query tensor of shape (..., queries, d_k)
            Can have arbitrary leading batch dimensions
        K (torch.Tensor): Key tensor of shape (..., keys, d_k)
            Same leading dimensions as Q, same d_k dimension
        V (torch.Tensor): Value tensor of shape (..., keys, d_v)
            Same leading dimensions and sequence length as K
        mask (torch.Tensor | None): Boolean mask of shape (..., queries, keys)
            True means attend (information flows), False means mask out
            
    Returns:
        torch.Tensor: Attention output of shape (..., queries, d_v)
            Weighted combination of values based on query-key similarities
    """
    # Get the key dimension for scaling
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores Q @ K⊤
    # Q shape: (..., queries, d_k)
    # K shape: (..., keys, d_k)  
    # We need K⊤ with shape: (..., d_k, keys)
    # Use einops for clearer transpose operation
    K_transposed = rearrange(K, '... keys d_k -> ... d_k keys')
    # scores shape: (..., queries, keys)
    scores = Q @ K_transposed
    
    # Step 2: Scale by √d_k for numerical stability
    # This prevents the softmax from saturating when d_k is large
    # The scaling ensures the variance of the dot products remains manageable
    scores = scores / (d_k ** 0.5)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        # For masked positions (False values), set scores to -∞
        # This ensures that after softmax, these positions have probability 0
        # We use a large negative value instead of true -∞ for numerical stability
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Step 4: Apply softmax to get attention weights
    # This converts the scaled scores into a probability distribution
    # Each row (query) sums to 1.0 across all keys it can attend to
    attention_weights = run_softmax(scores, dim=-1)  # Shape: (..., queries, keys)
    
    # Step 5: Apply attention weights to values
    # This computes the weighted combination of values for each query
    # attention_weights shape: (..., queries, keys)
    # V shape: (..., keys, d_v)
    # output shape: (..., queries, d_v)
    output = attention_weights @ V
    
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention as described in Vaswani et al. [2017].
    
    Mathematical formulation:
    MultiHeadSelfAttention(x) = W_O * MultiHead(W_Q*x, W_K*x, W_V*x)
    
    Where MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
    And head_i = Attention(Q_i, K_i, V_i) for slice i of Q, K, V
    
    Key concepts you'll implement:
    1. Linear projections: W_Q, W_K, W_V (shape: d_model -> num_heads * head_dim)
    2. Head splitting: Reshape to separate heads 
    3. RoPE application: Apply to Q and K (not V)
    4. Causal masking: Prevent attending to future tokens
    5. Scaled dot-product attention: For each head independently
    6. Head concatenation: Combine all head outputs
    7. Output projection: W_O to get final result
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        
        # Store d_model, num_heads as instance variables
        # Calculate head_dim = d_model // num_heads (this is d_k = d_v from the paper)
        # Assert that d_model % num_heads == 0 to ensure even division
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        assert d_model % self.num_heads == 0, f"d_model must be divisible by num_heads"
        self.head_dim = d_model // self.num_heads
        
        # W_Q: Projects input to query vectors for ALL heads (shape: d_model -> d_model)
        # W_K: Projects input to key vectors for ALL heads (shape: d_model -> d_model) 
        # W_V: Projects input to value vectors for ALL heads (shape: d_model -> d_model)
        # W_O: Output projection after concatenating heads (shape: d_model -> d_model)
        # Example: self.W_Q = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_Q = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_K = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_V = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_O = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        
        # Only create if max_seq_len is not None
        # Use your RotaryPositionalEmbedding class
        # Apply to queries and keys (head_dim must be even for RoPE)
        # Example: self.rope = RotaryPositionalEmbedding(theta=theta, d_k=head_dim, max_seq_len=max_seq_len, device=device)
        if self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Token positions for RoPE of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply W_Q, W_K, W_V to input x
        Q = self.W_Q(x)  # Shape: (batch_size, seq_len, d_model)
        K = self.W_K(x)  # Shape: (batch_size, seq_len, d_model)
        V = self.W_V(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Transform from (batch seq d_model) to (batch heads seq head_dim)
        # Use einops.rearrange for self-documenting transformations:
        Q = rearrange(Q, 'batch seq (heads head_dim) -> batch heads seq head_dim',
                      heads=self.num_heads, head_dim=self.head_dim)
        K = rearrange(K, 'batch seq (heads head_dim) -> batch heads seq head_dim',
                      heads=self.num_heads, head_dim=self.head_dim)
        V = rearrange(V, 'batch seq (heads head_dim) -> batch heads seq head_dim',
                      heads=self.num_heads, head_dim=self.head_dim)
        
        # Only if self.rope exists and token_positions is provided
        if self.rope is not None and token_positions is not None:
            # RoPE treats heads as batch dimensions, so reshape for RoPE application:
            #
            # Method: Flatten batch and head dimensions for RoPE, then reshape back
            Q_flat = rearrange(Q, 'batch heads seq head_dim -> (batch heads) seq head_dim')
            K_flat = rearrange(K, 'batch heads seq head_dim -> (batch heads) seq head_dim')

            # Expand token positions for all heads using einops
            positions_expanded = repeat(token_positions, 'batch seq -> batch heads seq', heads=self.num_heads)
            positions_flat = rearrange(positions_expanded, 'batch heads seq -> (batch heads) seq')

            # Apply RoPE (same rotation for all heads)
            Q_rotated = self.rope(Q_flat, positions_flat)
            K_rotated = self.rope(K_flat, positions_flat)

            # Reshape back to multi-head format using einops
            Q = rearrange(Q_rotated, '(batch heads) seq head_dim -> batch heads seq head_dim',
                          batch=batch_size, heads=self.num_heads)
            K = rearrange(K_rotated, '(batch heads) seq head_dim -> batch heads seq head_dim',
                          batch=batch_size, heads=self.num_heads)
        # Note: V is not rotated with RoPE since values don't need positional information
        
        # Shape should be (seq_len, seq_len) 
        # mask[i][j] = True if token i CAN attend to token j (i.e., j <= i)
        # mask[i][j] = False if token i CANNOT attend to token j (i.e., j > i)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        # Method 2: Use broadcasted comparison with torch.arange (more efficient, but hard to read)
        
        # Use your scaled_dot_product_attention function from earlier
        # Input shapes: Q, K, V are (batch_size, num_heads, seq_len, head_dim)
        # Mask shape: (seq_len, seq_len) - will broadcast to all batch and head dimensions
        # Output shape: (batch_size, num_heads, seq_len, head_dim)
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        
        # Transform from (batch heads seq head_dim) back to (batch seq d_model)
        # This "concatenates" the heads by flattening the head dimension
        # Use einops.rearrange for self-documenting transformation:
        attn_output = rearrange(attn_output, 'batch heads seq head_dim -> batch seq (heads head_dim)')
        # einops handles the reshaping automatically and clearly shows the transformation
        
        # Apply output projection to get final result
        output = self.W_O(attn_output)
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        
        return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block as described in the assignment.
    
    Architecture (following Figure 2):
    1. First sublayer: x + MultiHeadSelfAttention(RMSNorm(x))
    2. Second sublayer: y + SwiGLU(RMSNorm(y))
    
    This is a "pre-norm" architecture where normalization happens BEFORE
    the main operation (attention/feedforward), not after.
    
    Args:
        d_model (int): Dimensionality of the Transformer block inputs/outputs
        num_heads (int): Number of heads for multi-head self-attention
        d_ff (int): Dimensionality of the feed-forward inner layer
        max_seq_len (int): Maximum sequence length for RoPE (optional)
        theta (float): RoPE parameter (default: 10000.0)
        eps (float): RMSNorm epsilon for numerical stability (default: 1e-5)
        device: Device to place parameters on
        dtype: Data type for parameters
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = None, 
                 theta: float = 10000.0, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        # Store parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Create the two RMSNorm layers (one for each sublayer)
        # First RMSNorm layer applied before multi-head attention
        self.ln1 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        # Second RMSNorm layer applied before feed-forward network
        self.ln2 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        
        # Create the multi-head self-attention layer
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            theta=theta, 
            device=device, 
            dtype=dtype
        )
        
        # Create the feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply the pre-norm Transformer block.
        
        Architecture:
        1. First sublayer: x + MultiHeadSelfAttention(RMSNorm(x))
        2. Second sublayer: y + SwiGLU(RMSNorm(y))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Token positions for RoPE of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        
        # First sublayer: Multi-head self-attention with residual connection
        # Step 1: Apply RMSNorm to input
        normalized_input = self.ln1(x)
        
        # Step 2: Apply multi-head self-attention
        # Pass token_positions for RoPE if provided
        attn_output = self.attn(normalized_input, token_positions)
        
        # Step 3: Add residual connection
        # This gives us: y = x + MultiHeadSelfAttention(RMSNorm(x))
        y = x + attn_output
        
        # Second sublayer: Feed-forward network with residual connection  
        # Step 4: Apply RMSNorm to the output of first sublayer
        normalized_y = self.ln2(y)
        
        # Step 5: Apply feed-forward network (SwiGLU)
        ffn_output = self.ffn(normalized_y)
        
        # Step 6: Add residual connection
        # This gives us: final_output = y + SwiGLU(RMSNorm(y))
        final_output = y + ffn_output
        
        return final_output


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model as described in Section 3.1 and Figure 1.
    
    Architecture:
    1. Token embeddings: Convert token IDs to dense vectors
    2. Multiple Transformer blocks: Stack of self-attention and feed-forward layers
    3. Final layer norm: Normalize the output of the last transformer block
    4. Language model head: Project to vocabulary size
    5. Softmax: Convert logits to probability distribution over vocabulary
    
    This implements a causal (autoregressive) language model that predicts the next token
    given the previous tokens in the sequence.
    
    Args:
        vocab_size (int): Size of the vocabulary (number of possible tokens)
        context_length (int): Maximum sequence length the model can process
        d_model (int): Dimensionality of the model embeddings and sublayer outputs
        num_layers (int): Number of Transformer blocks to stack
        num_heads (int): Number of attention heads in each Transformer block
        d_ff (int): Dimensionality of the feed-forward inner layer
        rope_theta (float): RoPE base frequency parameter (default: 10000.0)
        eps (float): RMSNorm epsilon for numerical stability (default: 1e-5)
        device: Device to place parameters on
        dtype: Data type for parameters
    """
    
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, rope_theta: float = 10000.0, eps: float = 1e-5, 
                 device=None, dtype=None):
        super().__init__()
        
        # Store parameters
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        # Token embeddings: Convert token IDs to dense vectors
        # Maps vocab_size token IDs to d_model dimensional embeddings
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            device=device, 
            dtype=dtype
        )
        
        # Stack of Transformer blocks
        # Each block applies self-attention and feed-forward transformation
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                eps=eps,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization applied to the output of the last transformer block
        # This is common in modern transformer architectures for stability
        self.ln_final = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        
        # Language model head: Projects final hidden states to vocabulary logits
        # This is typically a linear layer that outputs unnormalized probabilities
        # over the vocabulary for next token prediction
        self.lm_head = Linear(
            in_features=d_model, 
            out_features=vocab_size, 
            device=device, 
            dtype=dtype
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Language Model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length)
                Each value should be an integer in range [0, vocab_size-1]
                
        Returns:
            torch.Tensor: Logits over vocabulary of shape (batch_size, sequence_length, vocab_size)
                Unnormalized scores for next token prediction at each position
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Convert token IDs to embeddings
        # input_ids shape: (batch_size, seq_len)
        # embeddings shape: (batch_size, seq_len, d_model)
        x = self.token_embeddings(input_ids)
        
        # Step 2: Generate token positions for RoPE
        # Positions are [0, 1, 2, ..., seq_len-1] for each sequence in the batch
        token_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Step 3: Pass through all Transformer blocks
        # Each block applies: x + MultiHeadSelfAttention(RMSNorm(x)) followed by x + SwiGLU(RMSNorm(x))
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Step 4: Apply final layer normalization
        # This stabilizes training and is standard in modern transformer architectures
        x = self.ln_final(x)
        
        # Step 5: Project to vocabulary size for next token prediction
        # x shape: (batch_size, seq_len, d_model)
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        # Step 6: Return logits (unnormalized scores)
        # In practice, we typically return logits and apply softmax separately
        # This is for numerical stability during training with cross-entropy loss
        # logits shape: (batch_size, seq_len, vocab_size)
        return logits


class Linear(nn.Module):
    """
    A Linear transformation module that applies a linear transformation to input data.
    
    This class implements a linear layer (also known as a fully connected layer or dense layer)
    that performs the operation: output = input @ W.T, where W is a learnable weight matrix.
    
    Unlike PyTorch's built-in nn.Linear, this implementation does not include a bias term.
    
    Args:
        in_features (int): Size of each input sample (number of input features)
        out_features (int): Size of each output sample (number of output features)  
        device (torch.device | None): Device to place the parameters on (CPU/GPU)
        dtype (torch.dtype | None): Data type for the parameters (float32, float64, etc.)
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        # Call the parent class constructor to properly initialize the nn.Module
        # This sets up important internal state for parameter tracking and gradient computation
        super().__init__()
        
        # Store the dimensions for reference and validation
        self.in_features = in_features
        self.out_features = out_features
        
        # Create the weight parameter W with shape (out_features, in_features)
        # We store W (not W.T) for memory ordering efficiency as mentioned in instructions
        # nn.Parameter wraps a tensor and tells PyTorch this is a learnable parameter
        # that should be included in gradients and optimizer updates
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # Initialize the weights using truncated normal distribution
        # This initialization helps with stable training and prevents vanishing/exploding gradients
        # trunc_normal_ initializes values from a truncated normal distribution
        torch.nn.init.trunc_normal_(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input tensor.
        
        The mathematical operation performed is: output = x @ W.T
        where @ denotes matrix multiplication and W.T is the transpose of W.
        
        For input shape (..., in_features), the output shape will be (..., out_features).
        The ... represents arbitrary leading dimensions (batch size, sequence length, etc.)
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)
                The last dimension must match self.in_features
                
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
                The transformation result with the same leading dimensions as input
        """
        # Perform matrix multiplication: x @ W.T
        # x has shape (..., in_features) 
        # W has shape (out_features, in_features)
        # W.T has shape (in_features, out_features)
        # Result has shape (..., out_features)
        return x @ self.W.T


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Test adapter function for the Linear module.
    
    This function creates a Linear layer with the specified dimensions,
    loads the provided weights into it, and applies it to the input features.
    
    The purpose is to test that our Linear implementation can correctly
    load pre-trained weights and produce the expected output transformations.

    Args:
        d_in (int): The size of the input dimension (number of input features)
        d_out (int): The size of the output dimension (number of output features)  
        weights (Float[Tensor, "d_out d_in"]): Pre-trained linear weights to load
            Shape is (d_out, d_in) matching our Linear class's W parameter shape
        in_features (Float[Tensor, "... d_in"]): Input tensor to transform
            Can have arbitrary leading dimensions (...) for batching

    Returns:
        Float[Tensor, "... d_out"]: The transformed output tensor
            Same leading dimensions as input, but last dimension is d_out
    """
    # Create an instance of our Linear module with the specified dimensions
    # We don't specify device/dtype, so it will use defaults (CPU, float32)
    linear_layer = Linear(in_features=d_in, out_features=d_out)
    
    # Load the provided weights into our linear layer
    # The weights tensor should have shape (d_out, d_in) which matches our W parameter
    # We use .data to directly assign without affecting gradients (since this is testing)
    linear_layer.W.data = weights
    
    # Apply the linear transformation to the input features
    # This calls our forward method: in_features @ W.T
    # PyTorch automatically handles the broadcasting for batch dimensions
    output = linear_layer(in_features)
    
    return output


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Test adapter function for the Embedding module.
    
    This function creates an Embedding layer with the specified vocabulary size and
    embedding dimension, loads the provided weights into it, and applies it to the
    input token IDs to retrieve the corresponding embedding vectors.
    
    The purpose is to test that our Embedding implementation can correctly
    load pre-trained embedding weights and produce the expected embedding lookups.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary (num_embeddings)
        d_model (int): The size of the embedding dimension (embedding_dim)
        weights (Float[Tensor, "vocab_size d_model"]): Pre-trained embedding weights to load
            Shape is (vocab_size, d_model) matching our Embedding class's weight parameter shape
        token_ids (Int[Tensor, "..."]): Input tensor of token IDs to look up embeddings for
            Can have arbitrary leading dimensions (...) for batching and sequence length

    Returns:
        Float[Tensor, "... d_model"]: The embedding vectors for the input token IDs
            Same leading dimensions as input, with d_model as the final dimension
    """
    # Create an instance of our Embedding module with the specified dimensions
    # We don't specify device/dtype, so it will use defaults (CPU, float32)
    embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    # Load the provided weights into our embedding layer
    # The weights tensor should have shape (vocab_size, d_model) which matches our weight parameter
    # We use .data to directly assign without affecting gradients (since this is testing)
    embedding_layer.weight.data = weights
    
    # Apply the embedding lookup to the input token IDs
    # This calls our forward method which performs: weight[token_ids]
    # PyTorch automatically handles the broadcasting for batch and sequence dimensions
    output = embedding_layer(token_ids)
    
    return output


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Test adapter function for the SwiGLU feed-forward network.
    
    This function creates a SwiGLU network with the specified dimensions,
    loads the provided weights into the three linear layers (W1, W2, W3),
    and applies the SwiGLU transformation to the input features.
    
    The purpose is to test that our SwiGLU implementation can correctly
    load pre-trained weights and produce the expected feed-forward output.

    Args:
        d_model (int): Dimensionality of the feedforward input and output (hidden size).
        d_ff (int): Dimensionality of the intermediate projection (typically ~8/3 * d_model).
        w1_weight (Float[Tensor, "d_ff d_model"]): Pre-trained weights for W1 (gate projection)
            Shape is (d_ff, d_model) - projects input to intermediate dimension
        w2_weight (Float[Tensor, "d_model d_ff"]): Pre-trained weights for W2 (output projection)
            Shape is (d_model, d_ff) - projects intermediate back to output dimension
        w3_weight (Float[Tensor, "d_ff d_model"]): Pre-trained weights for W3 (value projection)  
            Shape is (d_ff, d_model) - projects input to intermediate dimension
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer
            Can have arbitrary leading dimensions (...) for batching and sequence length

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings with same shape as input
            The result of applying SwiGLU: W2(SiLU(W1(x)) ⊙ W3(x))
    """
    # Create an instance of our SwiGLU feed-forward network
    # We don't specify device/dtype, so it will use defaults (CPU, float32)
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    # Load the provided weights into the three linear layers
    # Each weight tensor matches the expected shape for the corresponding layer
    
    # W1 (gate projection): d_model -> d_ff, so weight shape is (d_ff, d_model)
    swiglu.w1.W.data = w1_weight
    
    # W2 (output projection): d_ff -> d_model, so weight shape is (d_model, d_ff)  
    swiglu.w2.W.data = w2_weight
    
    # W3 (value projection): d_model -> d_ff, so weight shape is (d_ff, d_model)
    swiglu.w3.W.data = w3_weight
    
    # Apply the SwiGLU transformation to the input features
    # This calls our forward method which performs:
    # 1. Gate path: SiLU(W1(x))
    # 2. Value path: W3(x) 
    # 3. Gating: SiLU(W1(x)) * W3(x)
    # 4. Output: W2(gated_result)
    output = swiglu(in_features)
    
    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Test adapter function for scaled dot-product attention.
    
    This function directly applies the scaled dot-product attention mechanism
    to the provided query, key, and value tensors, with optional masking.
    
    The purpose is to test that our attention implementation correctly
    computes the attention weights and produces the expected output.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
            Shape: (..., num_queries, d_k) where ... represents batch dimensions
        K (Float[Tensor, " ... keys d_k"]): Key tensor  
            Shape: (..., num_keys, d_k) with same batch dimensions as Q
        V (Float[Tensor, " ... values d_v"]): Values tensor
            Shape: (..., num_keys, d_v) with same batch and sequence dimensions as K
        mask (Float[Tensor, " ... queries keys"] | None): Optional boolean mask
            Shape: (..., num_queries, num_keys) 
            True = attend to this position, False = mask out this position
            
    Returns:
        Float[Tensor, " ... queries d_v"]: Attention output
            Shape: (..., num_queries, d_v) - same batch dimensions as input
            Weighted combination of values based on query-key attention
    """
    # Apply scaled dot-product attention using our implementation
    # This performs the full attention computation:
    # 1. Compute Q @ K⊤ and scale by √d_k
    # 2. Apply mask (set False positions to -∞)  
    # 3. Apply softmax to get attention weights
    # 4. Multiply weights by V to get final output
    output = scaled_dot_product_attention(Q, K, V, mask)
    
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Test adapter function for multi-head self-attention.
    
    This function creates a MultiHeadSelfAttention module, loads the provided
    weights into the projection layers, and applies multi-head attention to
    the input features with causal masking.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        q_proj_weight (Float[Tensor, "d_model d_in"]): Weights for the Q projection
            Shape: (d_model, d_in) - projects input to query vectors for all heads
        k_proj_weight (Float[Tensor, "d_model d_in"]): Weights for the K projection  
            Shape: (d_model, d_in) - projects input to key vectors for all heads
        v_proj_weight (Float[Tensor, "d_model d_in"]): Weights for the V projection
            Shape: (d_model, d_in) - projects input to value vectors for all heads
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
            Shape: (d_model, d_model) - projects concatenated heads back to d_model
        in_features (Float[Tensor, "... sequence_length d_in"]): Input tensor
            Can have arbitrary batch dimensions, last two dims are (seq_len, d_in)

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Output of multi-head attention
            Same shape as input but last dimension is d_model
    """
    
    # Use the MultiHeadSelfAttention class you implemented above
    # Don't pass max_seq_len since this test doesn't use RoPE
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    
    # The test provides pre-trained weights that need to be loaded into your model
    # Load weights using .data attribute to avoid affecting gradients
    mha.W_Q.W.data = q_proj_weight  # Query projection weights
    mha.W_K.W.data = k_proj_weight  # Key projection weights
    mha.W_V.W.data = v_proj_weight  # Value projection weights
    mha.W_O.W.data = o_proj_weight  # Output projection weights
    
    # Call the forward method of your MultiHeadSelfAttention module
    # Don't pass token_positions since this test doesn't use RoPE
    output = mha(in_features)
    
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Test adapter function for the RotaryPositionalEmbedding module.
    
    This function creates a RoPE module with the specified parameters,
    and applies rotary positional embedding to the input query or key tensor.
    
    The purpose is to test that our RoPE implementation can correctly
    apply pairwise rotations based on token positions and produce the
    expected positionally-encoded output.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
            Must be even since RoPE works on pairs of dimensions.
        theta (float): RoPE base frequency parameter Θ (typically 10000).
        max_seq_len (int): Maximum sequence length for precomputing cos/sin values.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to apply RoPE to.
            Can be query or key vectors with arbitrary leading batch dimensions.
        token_positions (Int[Tensor, "... sequence_length"]): Token position indices.
            Specifies the position of each token in the sequence for rotation calculation.
            
    Returns:
        Float[Tensor, "... sequence_length d_k"]: Tensor with RoPE applied.
            Same shape as input, with rotary positional encoding applied.
    """
    # Create an instance of our RotaryPositionalEmbedding module
    # We don't specify device, so it will use defaults (CPU)
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    
    # Apply RoPE rotation to the input query or key tensor
    # This calls our forward method which:
    # 1. Extracts cos/sin values for given token positions
    # 2. Reshapes input to separate dimension pairs
    # 3. Applies 2D rotations to each pair: 
    #    rotated_even = x_even * cos - x_odd * sin
    #    rotated_odd = x_even * sin + x_odd * cos
    # 4. Combines rotated pairs back to original shape
    output = rope(in_query_or_key, token_positions)
    
    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # Create a TransformerBlock instance with the specified parameters
    transformer = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    
    # Load the provided weights into the transformer block components
    # Load RMSNorm weights for both sublayers
    transformer.ln1.weight.data = weights["ln1.weight"]
    transformer.ln2.weight.data = weights["ln2.weight"]
    
    # Load multi-head attention weights
    transformer.attn.W_Q.W.data = weights["attn.q_proj.weight"]
    transformer.attn.W_K.W.data = weights["attn.k_proj.weight"]
    transformer.attn.W_V.W.data = weights["attn.v_proj.weight"]
    transformer.attn.W_O.W.data = weights["attn.output_proj.weight"]
    
    # Load SwiGLU feed-forward network weights
    transformer.ffn.w1.W.data = weights["ffn.w1.weight"]
    transformer.ffn.w2.W.data = weights["ffn.w2.weight"]
    transformer.ffn.w3.W.data = weights["ffn.w3.weight"]
    
    # Generate token positions for RoPE (0, 1, 2, ..., seq_len-1)
    batch_size, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    # Apply the transformer block
    return transformer.forward(in_features, token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # Create a TransformerLM instance with the specified parameters
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    
    # Load token embedding weights
    model.token_embeddings.weight.data = weights["token_embeddings.weight"]
    
    # Load weights for each transformer layer
    for layer_idx in range(num_layers):
        layer = model.layers[layer_idx]
        
        # Load RMSNorm weights for this layer
        layer.ln1.weight.data = weights[f"layers.{layer_idx}.ln1.weight"]
        layer.ln2.weight.data = weights[f"layers.{layer_idx}.ln2.weight"]
        
        # Load multi-head attention weights for this layer
        layer.attn.W_Q.W.data = weights[f"layers.{layer_idx}.attn.q_proj.weight"]
        layer.attn.W_K.W.data = weights[f"layers.{layer_idx}.attn.k_proj.weight"]
        layer.attn.W_V.W.data = weights[f"layers.{layer_idx}.attn.v_proj.weight"]
        layer.attn.W_O.W.data = weights[f"layers.{layer_idx}.attn.output_proj.weight"]
        
        # Load SwiGLU feed-forward network weights for this layer
        layer.ffn.w1.W.data = weights[f"layers.{layer_idx}.ffn.w1.weight"]
        layer.ffn.w2.W.data = weights[f"layers.{layer_idx}.ffn.w2.weight"]
        layer.ffn.w3.W.data = weights[f"layers.{layer_idx}.ffn.w3.weight"]
    
    # Load final layer norm weights
    model.ln_final.weight.data = weights["ln_final.weight"]
    
    # Load language model head weights
    model.lm_head.W.data = weights["lm_head.weight"]
    
    # Run forward pass
    output = model(in_indices)
    
    return output


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Test adapter function for the RMSNorm module.
    
    This function creates an RMSNorm layer with the specified dimensions and epsilon,
    loads the provided weights into it, and applies it to the input features to
    perform root mean square normalization.
    
    The purpose is to test that our RMSNorm implementation can correctly
    load pre-trained normalization weights and produce the expected normalized output.

    Args:
        d_model (int): The dimensionality of the RMSNorm input (hidden dimension).
        eps (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): Pre-trained RMSNorm scale weights to load
            Shape is (d_model,) matching our RMSNorm class's weight parameter shape
        in_features (Float[Tensor, "... d_model"]): Input features to normalize
            Can have arbitrary leading dimensions (...) for batching and sequence length
            The last dimension must be d_model

    Returns:
        Float[Tensor, "... d_model"]: Tensor with the same shape as `in_features` 
            containing the RMSNorm-normalized output
    """
    # Create an instance of our RMSNorm module with the specified dimensions and epsilon
    # We don't specify device/dtype, so it will use defaults (CPU, float32)
    rmsnorm_layer = RMSNorm(d_model=d_model, eps=eps)
    
    # Load the provided weights into our RMSNorm layer
    # The weights tensor should have shape (d_model,) which matches our weight parameter
    # We use .data to directly assign without affecting gradients (since this is testing)
    rmsnorm_layer.weight.data = weights
    
    # Apply RMSNorm to the input features
    # This calls our forward method which performs:
    # 1. Upcast to float32 for numerical stability
    # 2. Compute RMS: sqrt(mean(x^2) + eps)
    # 3. Normalize: x / RMS
    # 4. Scale: normalized * weight
    # 5. Downcast back to original dtype
    output = rmsnorm_layer(in_features)
    
    return output


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    Apply SiLU (Sigmoid Linear Unit) activation function element-wise to the input tensor.
    
    SiLU is also known as Swish activation function. It's defined as:
    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    
    This activation function is smooth, non-monotonic, and has been shown to work well
    in deep networks, particularly in transformer models.

    Args:
        in_features (Float[Tensor, "..."]): Input features to apply SiLU activation to.
            Can have arbitrary shape - the activation is applied element-wise.

    Returns:
        Float[Tensor, "..."]: Output tensor with same shape as input, with SiLU applied
        to each element.
    """
    # SiLU(x) = x * sigmoid(x)
    # We use torch.sigmoid for numerical stability as recommended in the instructions
    # sigmoid(x) = 1 / (1 + exp(-x)) - this is computed stably by torch.sigmoid
    # 
    # The multiplication x * sigmoid(x) gives us the SiLU activation:
    # - For large positive x: sigmoid(x) ≈ 1, so SiLU(x) ≈ x (linear behavior)
    # - For large negative x: sigmoid(x) ≈ 0, so SiLU(x) ≈ 0 (saturates to zero)
    # - Around x = 0: provides smooth transition with some negative values allowed
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    import numpy as np
    import torch
    
    # Calculate the maximum valid starting position
    # We need context_length tokens for input and 1 more for the target
    max_start_idx = len(dataset) - context_length
    
    # Sample random starting positions for each sequence in the batch
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Initialize arrays to store input sequences and targets
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)
    
    # Extract sequences for each batch element
    for i, start_idx in enumerate(start_indices):
        # Input sequence: x[start_idx:start_idx+context_length]
        inputs[i] = dataset[start_idx:start_idx + context_length]
        # Target sequence: x[start_idx+1:start_idx+context_length+1] (next tokens)
        targets[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and place on specified device
    input_tensor = torch.from_numpy(inputs).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    
    return input_tensor, target_tensor


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Apply softmax operation to a specified dimension of the input tensor with numerical stability.
    
    Softmax converts unnormalized scores to a probability distribution:
    softmax(v)_i = exp(v_i) / Σ_j exp(v_j)
    
    For numerical stability, we use the mathematically equivalent but stable formulation:
    softmax(v)_i = exp(v_i - max(v)) / Σ_j exp(v_j - max(v))
    
    This prevents overflow by ensuring the maximum exponent is 0, since:
    exp(v_i - max(v)) ≤ exp(0) = 1

    Args:
        in_features (Float[Tensor, "..."]): Input tensor to apply softmax to.
            Can have arbitrary shape - softmax is applied along the specified dimension.
        dim (int): Dimension to apply softmax along.
            Must be a valid dimension index for the input tensor.

    Returns:
        Float[Tensor, "..."]: Output tensor with same shape as input.
            Values along the specified dimension form probability distributions (sum to 1).
    """
    # Step 1: Find the maximum value along the specified dimension for numerical stability
    # keepdim=True preserves the dimension so we can broadcast for subtraction
    # This max value will be subtracted from all elements to prevent overflow
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]  # Shape: same as input except dim has size 1
    
    # Step 2: Subtract the maximum value for numerical stability
    # This ensures the largest value becomes 0, preventing exp() from overflowing
    # Since softmax(v + c) = softmax(v) for any constant c, this doesn't change the result
    stable_input = in_features - max_vals  # Shape: same as input
    
    # Step 3: Apply the exponential function
    # Now all values are ≤ 0, so exp(values) ≤ 1, preventing overflow
    exp_values = torch.exp(stable_input)  # Shape: same as input
    
    # Step 4: Sum the exponential values along the specified dimension
    # keepdim=True preserves the dimension for broadcasting in the division step
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)  # Shape: same as input except dim has size 1
    
    # Step 5: Normalize by dividing each exp value by the sum
    # This creates a probability distribution along the specified dimension
    softmax_output = exp_values / sum_exp  # Shape: same as input
    
    # Verify the mathematical properties:
    # 1. All values are ≥ 0 (since exp() is always positive)
    # 2. Values along dim sum to 1 (due to normalization)
    # 3. Larger input values get larger probabilities (monotonic property preserved)
    
    return softmax_output


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Use PyTorch's built-in cross entropy loss function
    # F.cross_entropy applies softmax and computes negative log likelihood automatically
    # and averages across the batch by default
    return F.cross_entropy(inputs, targets)



def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Convert to list to allow multiple iterations if needed
    parameters = list(parameters)
    
    # Step 1: Collect all gradients that exist (skip parameters without gradients)
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    
    # If no gradients exist, nothing to clip
    if not gradients:
        return
    
    # Step 2: Compute the total L2 norm of all gradients combined
    # This is ||g||₂ where g is the concatenation of all parameter gradients
    total_norm = 0.0
    for grad in gradients:
        # Add the squared norm of this gradient tensor to total
        # grad.norm() computes the L2 norm of the tensor
        param_norm = grad.norm()
        total_norm += param_norm.item() ** 2
    
    # Take square root to get the L2 norm
    total_norm = total_norm ** 0.5
    
    # Step 3: Check if clipping is needed
    # Add epsilon for numerical stability (PyTorch default: 1e-6)
    eps = 1e-6
    
    if total_norm > max_l2_norm:
        # Step 4: Compute clipping factor
        # We want to scale all gradients by: max_l2_norm / (total_norm + eps)
        # This ensures the resulting norm will be just under max_l2_norm
        clip_factor = max_l2_norm / (total_norm + eps)
        
        # Step 5: Apply clipping to all gradients in-place
        for grad in gradients:
            # Scale each gradient by the clip factor
            # This modifies the gradient tensors in-place
            grad.mul_(clip_factor)
    
    # If total_norm <= max_l2_norm, we don't modify the gradients (no clipping needed)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    # Import our AdamW implementation
    import sys
    from pathlib import Path
    
    # Add the project root to the path to import our adamw_optimizer
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from adamw_optimizer import AdamW
    
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    import math
    
    # The cosine annealing schedule has three phases:
    # 1. Warmup phase: Linear increase from 0 to max_learning_rate
    # 2. Cosine annealing phase: Cosine decay from max_learning_rate to min_learning_rate  
    # 3. Post-annealing phase: Constant at min_learning_rate
    
    # Phase 1: Warmup (t < T_w)
    # Linear warmup: α_t = (t / T_w) * α_max
    if it < warmup_iters:
        # Linearly interpolate from 0 to max_learning_rate over warmup_iters steps
        # At t=0: learning_rate = 0
        # At t=T_w-1: learning_rate approaches α_max
        # At t=T_w: learning_rate = α_max (handled by next condition)
        return (it / warmup_iters) * max_learning_rate
    
    # Phase 2: Cosine Annealing (T_w ≤ t ≤ T_c)  
    # α_t = α_min + 0.5 * (1 + cos((t - T_w) / (T_c - T_w) * π)) * (α_max - α_min)
    elif it <= cosine_cycle_iters:
        # Calculate progress through the cosine cycle (0 to 1)
        # When t = T_w: progress = 0, cos(0) = 1, so α_t = α_max
        # When t = T_c: progress = 1, cos(π) = -1, so α_t = α_min
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        
        # Cosine annealing formula
        # cos(progress * π) goes from cos(0)=1 to cos(π)=-1
        # (1 + cos(...)) goes from 2 to 0
        # 0.5 * (1 + cos(...)) goes from 1 to 0
        # This creates smooth decay from α_max to α_min
        cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
        learning_rate = min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
        
        return learning_rate
    
    # Phase 3: Post-annealing (t > T_c)
    # Constant learning rate: α_t = α_min
    else:
        # After cosine cycle completes, maintain minimum learning rate
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    import torch
    
    # Create checkpoint dictionary containing all necessary state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # Save checkpoint to the specified output destination
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    import torch
    
    # Load checkpoint from the specified source
    checkpoint = torch.load(src)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the iteration number
    return checkpoint['iteration']


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    # Import our BPE tokenizer implementation
    import sys
    from pathlib import Path
    
    # Add the project root to the path to import our bpe_tokenizer
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from bpe_tokenizer import Tokenizer
    
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Import our BPE implementation
    import sys
    from pathlib import Path
    
    # Add the project root to the path to import our bpe_tokenizer
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from bpe_tokenizer import train_bpe
    
    return train_bpe(str(input_path), vocab_size, special_tokens)
