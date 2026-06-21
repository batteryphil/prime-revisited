import torch
import sys

def main():
    ckpt_path = "prime_step_550.pt"
    print(f"Loading {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt['state_dict']
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    layer_stats = []
    
    # We are looking for layers that have .base_idx, .fine_idx, and .init_combined
    # The keys look like: backbone.layers.0.mixer.in_proj.base_idx
    prefix_set = set()
    for key in sd.keys():
        if key.endswith('.base_idx'):
            prefix = key[:-9] # remove '.base_idx'
            prefix_set.add(prefix)

    print(f"Found {len(prefix_set)} PRIME layers.")

    total_params_all = 0
    total_moved_all = 0

    for prefix in prefix_set:
        base_key = f"{prefix}.base_idx"
        fine_key = f"{prefix}.fine_idx"
        init_key = f"{prefix}.init_combined"

        if base_key not in sd or fine_key not in sd or init_key not in sd:
            continue

        base_idx = sd[base_key].to(torch.int32)
        fine_idx = sd[fine_key].to(torch.int32)
        init_combined = sd[init_key].to(torch.int32)

        current_combined = base_idx * 256 + fine_idx
        
        # Calculate how many parameters have a different index than their initialization
        moved_mask = (current_combined != init_combined)
        moved_count = moved_mask.sum().item()
        total_count = moved_mask.numel()
        
        mig_pct = (moved_count / total_count) * 100.0

        layer_stats.append({
            'layer': prefix,
            'mig_pct': mig_pct,
            'moved': moved_count,
            'total': total_count
        })
        
        total_params_all += total_count
        total_moved_all += moved_count

    # Sort by migration percentage (descending)
    layer_stats.sort(key=lambda x: x['mig_pct'], reverse=True)

    print("\n" + "="*80)
    print(f"{'Layer Name':<50} | {'Migration %':<12} | {'Moved / Total'}")
    print("="*80)
    for stat in layer_stats:
        print(f"{stat['layer']:<50} | {stat['mig_pct']:>7.2f}%     | {stat['moved']:>9} / {stat['total']}")

    print("="*80)
    if total_params_all > 0:
        overall_pct = (total_moved_all / total_params_all) * 100.0
        print(f"{'OVERALL NETWORK':<50} | {overall_pct:>7.2f}%     | {total_moved_all:>9} / {total_params_all}")

    # Aggregate by projection type
    proj_types = {}
    for stat in layer_stats:
        # e.g., backbone.layers.0.mixer.dt_proj -> dt_proj
        parts = stat['layer'].split('.')
        proj_type = parts[-1]
        if proj_type not in proj_types:
            proj_types[proj_type] = {'moved': 0, 'total': 0}
        proj_types[proj_type]['moved'] += stat['moved']
        proj_types[proj_type]['total'] += stat['total']

    print("\n" + "="*80)
    print("AGGREGATE BY PROJECTION TYPE")
    print("="*80)
    
    agg_stats = []
    for ptype, counts in proj_types.items():
        pct = (counts['moved'] / counts['total']) * 100.0
        agg_stats.append((ptype, pct, counts['moved'], counts['total']))
        
    agg_stats.sort(key=lambda x: x[1], reverse=True)
    for ptype, pct, moved, total in agg_stats:
        print(f"{ptype:<50} | {pct:>7.2f}%     | {moved:>9} / {total}")


if __name__ == '__main__':
    main()
