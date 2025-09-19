"""
Utility script to recursively convert all numpy data types to standard Python types.
This can be used to debug or fix JSON serialization issues.
"""

import numpy as np

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to standard Python types
    """
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    
    # Handle lists
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    
    # Handle tuples
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    
    # Convert numpy scalar types to Python scalar types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Return unchanged
    else:
        return obj

def deep_type_check(obj, path=""):
    """
    Recursively check types in a nested structure and print numpy types
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            curr_path = f"{path}.{key}" if path else key
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                print(f"Found numpy type at {curr_path}: {type(value)}")
            deep_type_check(value, curr_path)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            curr_path = f"{path}[{i}]"
            if isinstance(item, (np.integer, np.floating, np.bool_)):
                print(f"Found numpy type at {curr_path}: {type(item)}")
            deep_type_check(item, curr_path)

# Example usage
if __name__ == "__main__":
    # Example with numpy types
    example = {
        "int_value": np.int32(42),
        "float_value": np.float32(3.14),
        "bool_value": np.bool_(True),
        "array": np.array([1, 2, 3]),
        "nested": {
            "another_bool": np.bool_(False),
            "coords": [(np.int32(10), np.int32(20)), (np.int32(30), np.int32(40))]
        }
    }
    
    # Check for numpy types
    print("Checking for numpy types...")
    deep_type_check(example)
    
    # Convert all numpy types
    print("\nConverting numpy types...")
    converted = convert_numpy_types(example)
    
    # Verify conversion
    print("\nVerifying conversion...")
    deep_type_check(converted)
    
    # Test JSON serialization
    import json
    try:
        json_str = json.dumps(converted)
        print("\nJSON serialization successful!")
        print(f"JSON length: {len(json_str)} characters")
    except TypeError as e:
        print(f"\nJSON serialization failed: {e}")