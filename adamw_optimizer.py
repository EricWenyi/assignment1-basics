"""
AdamW Optimizer Implementation

This module implements the AdamW optimizer as described in Loshchilov and Hutter (2019).
AdamW is a variant of Adam that fixes weight decay to improve regularization.

The key difference from Adam is that weight decay is applied directly to the parameters
rather than being incorporated into the gradient, which leads to better regularization.
"""

import torch
from torch.optim import Optimizer
from typing import Any, Dict, Iterable


class AdamW(Optimizer):
    """
    AdamW optimizer implementation.
    
    AdamW modifies Adam by decoupling weight decay from the gradient-based update.
    This leads to better regularization and often better performance on deep learning tasks.
    
    Algorithm (from Loshchilov and Hutter, 2019):
    1. Compute gradient: g = ∇θ ℓ(θ; B_t)
    2. Update biased first moment: m = β₁m + (1-β₁)g  
    3. Update biased second moment: v = β₂v + (1-β₂)g²
    4. Compute bias-corrected learning rate: α_t = α * √(1-β₂^t) / (1-β₁^t)
    5. Update parameters: θ = θ - α_t * m / (√v + ε)
    6. Apply weight decay: θ = θ - αλθ
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (α in the algorithm) - default: 1e-3
            betas: Coefficients for computing running averages (β₁, β₂) - default: (0.9, 0.999)
            eps: Term added to denominator for numerical stability (ε) - default: 1e-8  
            weight_decay: Weight decay coefficient (λ) - default: 1e-2
        """
        # Validate hyperparameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        # Store hyperparameters in defaults dictionary (required by torch.optim.Optimizer)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # Call parent class constructor
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            # Enable gradients temporarily to compute loss
            with torch.enable_grad():
                loss = closure()
        
        # Iterate through parameter groups (allows different hyperparameters for different layers)
        for group in self.param_groups:
            # Extract hyperparameters for this group
            lr = group['lr']                    # α (learning rate)
            beta1, beta2 = group['betas']       # β₁, β₂ (moment decay rates)
            eps = group['eps']                  # ε (numerical stability)
            weight_decay = group['weight_decay'] # λ (weight decay rate)
            
            # Iterate through parameters in this group
            for param in group['params']:
                # Skip parameters without gradients
                if param.grad is None:
                    continue
                
                # Get gradient (this is 'g' in the algorithm)
                grad = param.grad.data
                
                # Initialize state for this parameter if not already done
                if param not in self.state:
                    self.state[param] = {
                        'step': 0,                              # Time step counter (t)
                        'exp_avg': torch.zeros_like(param),     # First moment estimate (m)
                        'exp_avg_sq': torch.zeros_like(param),  # Second moment estimate (v)
                    }
                
                # Get state for this parameter
                state = self.state[param]
                exp_avg = state['exp_avg']          # m (first moment)
                exp_avg_sq = state['exp_avg_sq']    # v (second moment)
                
                # Increment step counter
                state['step'] += 1
                step = state['step']                # t (current time step)
                
                # Update first moment estimate: m = β₁m + (1-β₁)g
                # This is an exponential moving average of the gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment estimate: v = β₂v + (1-β₂)g²  
                # This is an exponential moving average of the squared gradient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias correction terms
                # As t increases, these approach 1, removing the initialization bias
                bias_correction1 = 1 - beta1 ** step    # 1 - β₁^t
                bias_correction2 = 1 - beta2 ** step    # 1 - β₂^t
                
                # Compute bias-corrected step size: α_t = α * √(1-β₂^t) / (1-β₁^t)
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # Update parameters: θ = θ - α_t * m / (√v + ε)
                # The denominator (√v + ε) provides adaptive scaling based on gradient history
                param.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
                
                # Apply weight decay: θ = θ - αλθ  
                # This is applied AFTER the gradient update (key difference from Adam)
                # Weight decay pulls parameters toward zero for regularization
                if weight_decay > 0:
                    param.data.mul_(1 - lr * weight_decay)
        
        return loss


# Alternative implementation using functional approach (more explicit)
class AdamWExplicit(Optimizer):
    """
    Alternative AdamW implementation with more explicit step-by-step computation.
    This version makes each algorithmic step very clear for educational purposes.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWExplicit, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # ===== STEP 1: Get current gradient =====
                grad = param.grad.data  # This is 'g' in algorithm
                
                # ===== STEP 2: Initialize state if needed =====
                if param not in self.state:
                    self.state[param] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(param.data),      # m vector
                        'exp_avg_sq': torch.zeros_like(param.data),   # v vector  
                    }
                
                state = self.state[param]
                m = state['exp_avg']        # First moment estimate
                v = state['exp_avg_sq']     # Second moment estimate
                
                # ===== STEP 3: Update time step =====
                state['step'] += 1
                t = state['step']
                
                # ===== STEP 4: Update first moment estimate =====
                # m = β₁ * m + (1 - β₁) * g
                beta1, beta2 = group['betas']
                m.mul_(beta1)
                m.add_(grad, alpha=(1 - beta1))
                
                # ===== STEP 5: Update second moment estimate =====  
                # v = β₂ * v + (1 - β₂) * g²
                v.mul_(beta2)
                v.addcmul_(grad, grad, value=(1 - beta2))
                
                # ===== STEP 6: Compute bias correction =====
                bias_corr1 = 1 - beta1 ** t
                bias_corr2 = 1 - beta2 ** t
                
                # ===== STEP 7: Compute adaptive learning rate =====
                # α_t = α * √(1-β₂^t) / (1-β₁^t)
                adapted_lr = group['lr'] * (bias_corr2 ** 0.5) / bias_corr1
                
                # ===== STEP 8: Update parameters =====
                # θ = θ - α_t * m / (√v + ε)
                denominator = v.sqrt().add_(group['eps'])
                param.data.addcdiv_(m, denominator, value=-adapted_lr)
                
                # ===== STEP 9: Apply weight decay =====
                # θ = θ - α * λ * θ
                if group['weight_decay'] > 0:
                    param.data.add_(param.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss