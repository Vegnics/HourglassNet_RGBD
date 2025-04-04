import keras
from keras.layers import Layer
"""
This piece of code was generated with help from OpenAI ChatGPT4o-mini to ease 
the transfer of weights from a baseline model to a model with attention mechanisms. 
"""

def print_layers_recursive(target_layer,path=""):
    # Determine sublayers (use fallback if needed)
    current_name = f"{path}/{target_layer.name}" if path else target_layer.name
    print(current_name)
    if hasattr(target_layer, 'layers') and target_layer.layers:
        tgt_sublayers = {l.name: l for l in target_layer.layers}
    else:
        tgt_sublayers = {
            k: v for k, v in vars(target_layer).items()
            if isinstance(v, Layer)
        }
    # Recurse through matched sublayers
    for name, tgt_sub in tgt_sublayers.items():
        print_layers_recursive(tgt_sub, path=current_name)
    return 

def recursive_weight_transfer(source_layer, target_layer, verbose=True, path=""):
    matched, skipped = 0, 0
    current_name = f"{path}/{target_layer.name}" if path else target_layer.name
    #if not path:
    #    print_layers_recursive(target_layer,path="")
    # Compare weights directly
    if source_layer.weights and target_layer.weights:
        sw = source_layer.get_weights()
        tw = target_layer.get_weights()

        if all(s.shape == t.shape for s, t in zip(sw, tw)):
            target_layer.set_weights(sw)
            matched += 1
            if verbose:
                print(f"[✓] Matched: {current_name}")
        else:
            skipped += 1
            if verbose:
                print(f"[!] Shape mismatch: {current_name}")
    elif verbose and (source_layer.weights or target_layer.weights):
        print(f"[!] Incomplete weights in: {current_name}")
        skipped += 1

    # Determine sublayers (use fallback if needed)
    if hasattr(target_layer, 'layers') and target_layer.layers:
        tgt_sublayers = {l.name: l for l in target_layer.layers}
        src_sublayers = {l.name: l for l in getattr(source_layer, 'layers', [])}
    else:
        tgt_sublayers = {
            k: v for k, v in vars(target_layer).items()
            if isinstance(v, Layer)
        }
        src_sublayers = {
            k: v for k, v in vars(source_layer).items()
            if isinstance(v, Layer)
        }

    # Recurse through matched sublayers
    for name, tgt_sub in tgt_sublayers.items():
        src_sub = src_sublayers.get(name)
        if src_sub:
            m, s = recursive_weight_transfer(src_sub, tgt_sub, verbose, path=current_name)
            matched += m
            skipped += s
        else:
            if verbose:
                print(f"[!] Missing sublayer: {current_name}/{name}")
            skipped += 1

    return matched, skipped