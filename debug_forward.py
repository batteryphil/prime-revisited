import torch
import torch.nn.functional as F
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear, CONFIG

def main():
    print("Initializing model...")
    model = TitanMIMO().cuda()
    lut = build_prime_lut()
    
    for layer in model.layers:
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
    for arm in model.mimo_arms:
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
    model.domain_router = PrimeLinear(model.domain_router, lut)
    
    ckpt_data = torch.load("/data/titan_checkpoints/titan_mimo_phase3_frozen.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt_data['state_dict'], strict=False)
    model = model.cuda()
    
    input_ids = torch.randint(0, 50000, (4, 256)).cuda()
    labels = input_ids.clone()
    domain_id = torch.tensor([0, 1, 2, 3]).cuda()
    
    # Run the exact logic from mamba3_titan_mimo.py
    x = model.embedding(input_ids)
    for layer in model.layers[:model._mid]:
        x = layer(x)
        
    router_logits = model.domain_router(F.layer_norm(x.detach(), (x.size(-1),)))
    route_weights = F.gumbel_softmax(router_logits, tau=1.0, hard=True, dim=-1)
    route_weights_detached = route_weights.detach()
    
    raw_arm_outs = [arm(x) for arm in model.mimo_arms]
    stacked = torch.stack(raw_arm_outs, dim=-1)
    collapsed = torch.einsum('b l d m, b l m -> b l d', stacked, route_weights_detached.to(stacked.dtype))
    x = x + collapsed
    
    for layer in model.layers[model._mid:]:
        x = layer(x)
        
    x = model.norm_f(x)
    logits = model.lm_head(x)
    
    # Calculate loss components
    logits_reshaped = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    labels_reshaped = labels[..., 1:].contiguous().view(-1)
    
    print("Logits Shape:", logits_reshaped.shape)
    print("Labels Shape:", labels_reshaped.shape)
    print("Logits Max:", logits_reshaped.max().item(), "Min:", logits_reshaped.min().item())
    
    ntp_loss = F.cross_entropy(logits_reshaped, labels_reshaped, ignore_index=0)
    print("NTP Loss (mean):", ntp_loss.item())
    print("NTP Loss (sum):", F.cross_entropy(logits_reshaped, labels_reshaped, ignore_index=0, reduction='sum').item())
    
    target = domain_id.unsqueeze(1).expand(-1, 256).long()
    routing_loss = F.cross_entropy(router_logits.view(-1, 4), target.reshape(-1))
    print("Routing Loss:", routing_loss.item())
    
    total = ntp_loss + 0.5 * routing_loss
    print("Total Loss:", total.item())

if __name__ == "__main__":
    main()
