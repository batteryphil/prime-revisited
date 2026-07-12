import re
import os

with open("/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime/mamba3_titan_mimo.py", "r") as f:
    code = f.read()

# 1. Update PrimeLinear.__init__
prime_init_find = """        # PI Controller Buffers
        self.register_buffer('dynamic_threshold', torch.tensor([12.0], dtype=torch.float32))
        self.register_buffer('pi_integral', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('last_flip_rate', torch.tensor([0.0], dtype=torch.float32))"""

prime_init_replace = """        import math
        self.target_flip_rate = kwargs.get('target_flip_rate', 0.15)
        
        # PI Controller Buffers (V2)
        self.register_buffer('log_threshold', torch.tensor([math.log(12.0)], dtype=torch.float32))
        self.register_buffer('pi_integral', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('ema_flip_rate', torch.tensor([0.0], dtype=torch.float32))
        
        # Telemetry Buffers
        self.register_buffer('last_grad_sign', torch.zeros(self.out_features, self.in_features, dtype=torch.int8))
        self.register_buffer('sign_agreement_ema', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('delta_log_thresh', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('step_counter', torch.tensor([0], dtype=torch.long))"""

# Fix def __init__ to accept **kwargs
code = code.replace("def __init__(self, module: nn.Linear, lut: torch.Tensor, name=\"unnamed\"):", "def __init__(self, module: nn.Linear, lut: torch.Tensor, name=\"unnamed\", **kwargs):")
code = code.replace(prime_init_find, prime_init_replace)

# 2. Update PrimeLinear._vote_hook
hook_find = """    def _vote_hook(self, grad):
        with torch.no_grad():
            # In-place operations to prevent intermediate tensor allocations and massive kernel overhead
            pressure = grad.sign().mul_(10).to(torch.int16)
            self.vote_buffer.add_(pressure).clamp_(-32760, 32760)"""

hook_replace = """    def _vote_hook(self, grad):
        with torch.no_grad():
            sign_t = grad.sign().to(torch.int8)
            
            # Compute sign agreement sample every 50 steps
            if self.step_counter[0] % 50 == 0:
                agreement = (sign_t == self.last_grad_sign).float().mean()
                self.sign_agreement_ema[0] = self.sign_agreement_ema[0] * 0.95 + agreement * 0.05
                
            self.last_grad_sign.copy_(sign_t)
            
            pressure = sign_t.mul_(10).to(torch.int16)
            self.vote_buffer.add_(pressure).clamp_(-32760, 32760)"""
code = code.replace(hook_find, hook_replace)

# 3. Update PrimeLinear.apply_votes authorization
auth_find = """        blocks    = flat_a.float().view(-1, bs)
        magnitude = blocks.abs().mean(dim=1, keepdim=True)
        authorized = (magnitude * (lr / 1e-4) >= self.dynamic_threshold[0])"""

auth_replace = """        import math
        blocks    = flat_a.float().view(-1, bs)
        magnitude = blocks.abs().mean(dim=1, keepdim=True)
        
        self.step_counter[0] += 1
        threshold = torch.exp(self.log_threshold[0])
        authorized = (magnitude * (lr / 1e-4) >= threshold)"""
code = code.replace(auth_find, auth_replace)

# 4. Update PrimeLinear PI Controller
pi_find = """        # PI Controller Update (Target 15% flip rate)
        with torch.no_grad():
            current_flip_rate = flips / n
            self.last_flip_rate[0] = current_flip_rate
            error = current_flip_rate - 0.15
            
            kp = 10.0
            ki = 0.1
            
            self.pi_integral[0] += error
            self.pi_integral[0] = torch.clamp(self.pi_integral[0], -50.0, 50.0)
            
            adjustment = (kp * error) + (ki * self.pi_integral[0])
            self.dynamic_threshold[0] = torch.clamp(self.dynamic_threshold[0] + adjustment, 2.0, 64.0)"""

pi_replace = """        # V2 Log-Space PI Controller
        with torch.no_grad():
            import math
            current_flip_rate = flips / max(n, 1)
            self.ema_flip_rate[0] = self.ema_flip_rate[0] * 0.999 + current_flip_rate * 0.001
            
            # Run controller every 50 steps, with a 1000 step warmup freeze
            if self.step_counter[0] > 1000 and self.step_counter[0] % 50 == 0:
                target = self.target_flip_rate
                error = (self.ema_flip_rate[0] - target) / max(target, 1e-3)
                
                kp = 0.5
                ki = 0.01
                
                self.pi_integral[0] += error
                adjustment = (kp * error) + (ki * self.pi_integral[0])
                
                new_log_thresh = self.log_threshold[0] + adjustment
                clamped_log = torch.clamp(new_log_thresh, math.log(2.0), math.log(128.0))
                
                self.delta_log_thresh[0] = clamped_log - self.log_threshold[0]
                self.log_threshold[0] = clamped_log
                
                # Anti-windup
                if self.log_threshold[0] >= math.log(128.0) or self.log_threshold[0] <= math.log(2.0):
                    self.pi_integral[0] = 0.0"""
code = code.replace(pi_find, pi_replace)
code = code.replace("vpu = (self.vote_buffer.float().abs().mean() / self.dynamic_threshold[0]).item()", "vpu = (self.vote_buffer.float().abs().mean() / threshold).item()")

# 5. Add outputs to the return dict in apply_votes
ret_find = """            'name': self.name, 'entropy': entropy,
            'vpu': vpu, 'reversals': reversals, 'numel': n,
            'retention_pressure': retention_pressure"""
ret_replace = """            'name': self.name, 'entropy': entropy,
            'vpu': vpu, 'reversals': reversals, 'numel': n,
            'retention_pressure': retention_pressure,
            'ema_flip': self.ema_flip_rate[0].item(),
            'sign_agreement': self.sign_agreement_ema[0].item(),
            'threshold': torch.exp(self.log_threshold[0]).item(),
            'delta_thresh': self.delta_log_thresh[0].item()"""
code = code.replace(ret_find, ret_replace)

# 6. Update ContinuousVoteOptimizer
opt_init_find = """                state['vote_buffer'] = torch.zeros_like(p, dtype=torch.int16)
                state['dynamic_threshold'] = torch.tensor(12.0, dtype=torch.float32, device=p.device)
                state['pi_integral'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['last_flip_rate'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)"""
opt_init_replace = """                import math
                state['vote_buffer'] = torch.zeros_like(p, dtype=torch.int16)
                state['log_threshold'] = torch.tensor(math.log(12.0), dtype=torch.float32, device=p.device)
                state['pi_integral'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['ema_flip_rate'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['delta_log_thresh'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['step_counter'] = torch.tensor(0, dtype=torch.long, device=p.device)
                state['target_flip_rate'] = 0.25  # LM head target"""
code = code.replace(opt_init_find, opt_init_replace)

opt_step_find = """            for p in group['params']:
                state = self.state[p]
                votes = state['vote_buffer']
                thresh = state['dynamic_threshold']
                
                authorized = votes.abs() >= thresh"""
opt_step_replace = """            for p in group['params']:
                import math
                state = self.state[p]
                votes = state['vote_buffer']
                state['step_counter'] += 1
                
                thresh = torch.exp(state['log_threshold'])
                authorized = votes.abs() >= thresh"""
code = code.replace(opt_step_find, opt_step_replace)

opt_pi_find = """                # PI Controller Update (Target 25% flip rate for continuous output layers)
                current_flip_rate = f / p.numel()
                state['last_flip_rate'].fill_(current_flip_rate)
                error = current_flip_rate - 0.25
                
                kp = 10.0
                ki = 0.1
                
                state['pi_integral'].add_(error).clamp_(-50.0, 50.0)
                adjustment = (kp * error) + (ki * state['pi_integral'])
                state['dynamic_threshold'].add_(adjustment).clamp_(2.0, 64.0)"""

opt_pi_replace = """                # V2 Log-Space PI Controller
                current_flip_rate = f / p.numel()
                state['ema_flip_rate'].mul_(0.999).add_(current_flip_rate * 0.001)
                
                if state['step_counter'].item() > 1000 and state['step_counter'].item() % 50 == 0:
                    target = state['target_flip_rate']
                    error = (state['ema_flip_rate'] - target) / max(target, 1e-3)
                    
                    kp, ki = 0.5, 0.01
                    state['pi_integral'].add_(error)
                    adjustment = (kp * error) + (ki * state['pi_integral'])
                    
                    new_log = state['log_threshold'] + adjustment
                    clamped_log = torch.clamp(new_log, math.log(2.0), math.log(128.0))
                    
                    state['delta_log_thresh'].copy_(clamped_log - state['log_threshold'])
                    state['log_threshold'].copy_(clamped_log)
                    
                    if clamped_log.item() >= math.log(128.0) or clamped_log.item() <= math.log(2.0):
                        state['pi_integral'].fill_(0.0)"""
code = code.replace(opt_pi_find, opt_pi_replace)

# 7. Update Model Init to set specific targets
init_loop_find = """    wrapped = 0
    for idx, layer in enumerate(model.layers):
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj,  lut, name=f"backbone_L{idx}_in_proj")
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut, name=f"backbone_L{idx}_out_proj")
        wrapped += 2
        
    arm_names = ['chat', 'math', 'code', 'tool']
    for idx, arm in enumerate(model.mimo_arms):
        aname = arm_names[idx] if idx < len(arm_names) else str(idx)
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj,  lut, name=f"mimo_arm_{aname}_in_proj")
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut, name=f"mimo_arm_{aname}_out_proj")
        wrapped += 2"""

init_loop_replace = """    wrapped = 0
    for idx, layer in enumerate(model.layers):
        # Specific target flip rates for hierarchy
        if idx < 7: target = 0.10
        elif idx < 21: target = 0.13
        else: target = 0.16
        
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj,  lut, name=f"backbone_L{idx}_in_proj", target_flip_rate=target)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut, name=f"backbone_L{idx}_out_proj", target_flip_rate=target)
        wrapped += 2
        
    arm_names = ['chat', 'math', 'code', 'tool']
    for idx, arm in enumerate(model.mimo_arms):
        aname = arm_names[idx] if idx < len(arm_names) else str(idx)
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj,  lut, name=f"mimo_arm_{aname}_in_proj", target_flip_rate=0.13)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut, name=f"mimo_arm_{aname}_out_proj", target_flip_rate=0.13)
        wrapped += 2"""
code = code.replace(init_loop_find, init_loop_replace)

# 8. Diagnostics
diag_find = """                print("[DIAGNOSTICS] Homeostatic Controller State:")
                
                # Discrete Layers
                for name, module in model.named_modules():
                    if hasattr(module, 'apply_votes'):
                        thresh = module.dynamic_threshold[0].item()
                        rate = module.last_flip_rate[0].item() * 100
                        print(f"  {name.rjust(30)} | thresh: {thresh:5.1f} | flip: {rate:4.1f}%")
                        
                # Continuous Layers
                print("\\n[DIAGNOSTICS] Continuous Layers (Optimizer):")
                for i, p in enumerate(optimizer.param_groups[0]['params']):
                    state = optimizer.state[p]
                    thresh = state['dynamic_threshold'].item()
                    rate = state['last_flip_rate'].item() * 100
                    name = f"continuous_param_{i}"
                    # Try to find its actual name
                    for n, param in model.named_parameters():
                        if param is p:
                            name = n
                            break
                    print(f"  {name.rjust(30)} | thresh: {thresh:5.1f} | flip: {rate:4.1f}%")"""

diag_replace = """                import math
                print("[DIAGNOSTICS] Homeostatic Controller State:")
                
                # Discrete Layers
                for name, module in model.named_modules():
                    if hasattr(module, 'apply_votes'):
                        thresh = math.exp(module.log_threshold[0].item())
                        rate = module.ema_flip_rate[0].item() * 100
                        target = module.target_flip_rate * 100
                        agmt = module.sign_agreement_ema[0].item() * 100
                        print(f"  {name.rjust(30)} | thresh: {thresh:5.1f} | EMA_flip: {rate:4.1f}% (T:{target:.0f}%) | Sign_Agmt: {agmt:4.1f}%")
                        
                # Continuous Layers
                print("\\n[DIAGNOSTICS] Continuous Layers (Optimizer):")
                for i, p in enumerate(optimizer.param_groups[0]['params']):
                    state = optimizer.state[p]
                    thresh = math.exp(state['log_threshold'].item())
                    rate = state['ema_flip_rate'].item() * 100
                    target = state['target_flip_rate'] * 100
                    name = f"continuous_param_{i}"
                    for n, param in model.named_parameters():
                        if param is p:
                            name = n
                            break
                    print(f"  {name.rjust(30)} | thresh: {thresh:5.1f} | EMA_flip: {rate:4.1f}% (T:{target:.0f}%)")"""
code = code.replace(diag_find, diag_replace)

with open("/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime/mamba3_titan_mimo.py", "w") as f:
    f.write(code)
print("mamba3_titan_mimo.py successfully patched for V2!")
