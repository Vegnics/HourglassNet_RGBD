import keras
from keras.layers import Layer
"""
This piece of code was generated with help from OpenAI ChatGPT4o-mini to ease 
the transfer of weights from a baseline model to a model with attention mechanisms. 
"""


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
    if source_layer.weights and target_layer.weights: #and target_layer.name != "ff_alpha":
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
"""

# Helper function to print layer structure (for debugging)
def print_layers_recursive(layer, indent=0, path=""):
    prefix = "  " * indent
    current_name = f"{path}/{layer.name}" if path else layer.name
    print(f"{prefix}- {current_name} ({type(layer).__name__})")
    for var in layer.variables:
        print(f"{prefix}    - {var.name} {var.shape}")

    # Determine sublayers
    if hasattr(layer, 'layers') and layer.layers:
        sublayers_iter = layer.layers
    else:
        # Fallback for layers that hold other layers as attributes
        sublayers_iter = [v for k, v in vars(layer).items() if isinstance(v, Layer)]

    for sub_layer in sublayers_iter:
        print_layers_recursive(sub_layer, indent + 1, current_name)


def recursive_weight_transfer(source_layer, target_layer, verbose=True, path=""):
    matched, skipped = 0, 0
    current_name = f"{path}/{target_layer.name}" if path else target_layer.name

    # --- Robust Weight Transfer by Variable Name ---
    if source_layer.variables and target_layer.variables:
        source_vars_map = {v.name: v for v in source_layer.variables}
        target_vars = target_layer.variables
        
        values_to_set = []
        all_shapes_match = True

        for target_var in target_vars:
            # Construct the expected source variable name.
            # This is tricky because TF appends numbers, e.g., "conv2d/kernel:0"
            # We need to match the "base name" without the trailing index ":0"
            # and potentially without the outer layer's name prefix if dealing with nested layers
            
            # Simplified approach: direct name match.
            # For complex cases, you might need to parse `target_var.name`
            # e.g., "layer_name/sub_layer_name/kernel:0" -> "sub_layer_name/kernel:0"
            
            # Common pattern: target_layer_name/variable_name:0
            # Source variable would be source_layer_name/variable_name:0
            # When comparing within the same recursion level, the `variable_name:0` part is crucial.
            
            # The most reliable way for source_var_name would be to strip target_layer.name and rebuild.
            # But usually, within a layer, TF assigns unique names like 'kernel:0', 'bias:0', 'moving_mean:0'
            # Let's try matching the full name for simplicity first.
            
            # Extract the unique part of the variable name after the layer's own name
            # Example: 'residual_block/conv_block/conv2d_1/kernel:0'
            # If target_layer.name is 'residual_block', we need 'conv_block/conv2d_1/kernel:0'
            
            # This logic assumes the structure and naming conventions are consistent between source and target
            # which is usually true if they are instances of the same class.
            
            # Determine the effective name without the root layer's prefix
            target_var_name_suffix = target_var.name.split(target_layer.name + '/')[-1]
            
            # Find the corresponding source variable by its suffix
            corresponding_source_var = None
            for sv_name, sv in source_vars_map.items():
                if sv_name.endswith(target_var_name_suffix):
                    corresponding_source_var = sv
                    break

            if corresponding_source_var is not None:
                if corresponding_source_var.shape == target_var.shape:
                    values_to_set.append(corresponding_source_var.numpy())
                else:
                    all_shapes_match = False
                    if verbose:
                        print(f"[!] Shape mismatch for variable: {target_var.name} (Source: {corresponding_source_var.shape}, Target: {target_var.shape})")
                    break # Break inner loop, cannot set weights for this layer
            else:
                all_shapes_match = False
                if verbose:
                    print(f"[!] Missing source variable: {target_var.name}")
                break # Break inner loop, cannot set weights for this layer

        if all_shapes_match and values_to_set: # Ensure we collected all values and no mismatch occurred
            try:
                # Use target_layer.set_weights() which expects values for target_layer.weights
                # It's safer to map to target_layer.weights explicitly than using its variables list directly.
                # Re-fetch source_layer.get_weights() for the matching order
                # This assumes get_weights() order is consistent IF names match
                
                # A more robust approach if full name matching failed:
                # build a list of numpy arrays in the order of target_layer.variables
                # using the source_vars_map.
                
                if len(values_to_set) == len(target_vars): # Double check we found all
                    target_layer.set_weights(values_to_set)
                    matched += 1
                    if verbose:
                        print(f"[✓] Matched variables for: {current_name}")
                else:
                    skipped += 1
                    if verbose:
                        print(f"[!] Partial variable match for: {current_name}. Expected {len(target_vars)}, found {len(values_to_set)}.")

            except Exception as e:
                skipped += 1
                if verbose:
                    print(f"[!] Error setting weights for {current_name}: {e}")
        else:
            skipped += 1
            if verbose:
                # This catches cases where source_layer.variables might be empty or shapes don't match initially
                if not source_layer.variables and target_layer.variables:
                    print(f"[!] Target has weights but source doesn't for: {current_name}")
                elif not target_layer.variables and source_layer.variables:
                     print(f"[!] Source has weights but target doesn't for: {current_name}")
                elif not target_layer.variables and not source_layer.variables:
                    # No weights expected, but not an error
                    pass
                else: # Catch all other mismatch scenarios
                    print(f"[!] Cannot transfer weights for: {current_name} due to variable mismatch.")
            
    elif verbose and (source_layer.variables or target_layer.variables):
        print(f"[!] Incomplete variables in: {current_name} (Source has: {bool(source_layer.variables)}, Target has: {bool(target_layer.variables)})")
        skipped += 1


    # Determine sublayers for recursion
    # Prioritize 'layers' attribute for Sequential/Model/Functional API layers
    # Fallback to checking attributes for custom Layer subclasses that might store sub-layers
    # as direct attributes (like your ConvBlockLayer, ResidualBlock).
    
    tgt_sublayers_dict = {}
    if hasattr(target_layer, 'layers') and isinstance(target_layer.layers, (list, tuple)):
        for l in target_layer.layers:
            if isinstance(l, Layer): # Ensure it's a Keras Layer
                tgt_sublayers_dict[l.name] = l
    else:
        for k, v in vars(target_layer).items():
            if isinstance(v, Layer):
                tgt_sublayers_dict[k] = v # Use attribute name as key

    src_sublayers_dict = {}
    if hasattr(source_layer, 'layers') and isinstance(source_layer.layers, (list, tuple)):
        for l in source_layer.layers:
            if isinstance(l, Layer):
                src_sublayers_dict[l.name] = l
    else:
        for k, v in vars(source_layer).items():
            if isinstance(v, Layer):
                src_sublayers_dict[k] = v

    # Recurse through matched sublayers by name
    for name, tgt_sub in tgt_sublayers_dict.items():
        src_sub = src_sublayers_dict.get(name)
        if src_sub:
            m, s = recursive_weight_transfer(src_sub, tgt_sub, verbose, path=current_name)
            matched += m
            skipped += s
        else:
            if verbose:
                print(f"[!] Missing sublayer in source for: {current_name}/{name}")
            skipped += 1

    return matched, skipped
