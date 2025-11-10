"""
Convert TFLite model to C byte array (Cell 15 from examples.ipynb).
This creates a C header file with the model embedded as a byte array for ESP32.
"""

import os

def hex_to_c_array(hex_data, var_name):
    """
    Convert binary data to C byte array format.
    
    Args:
        hex_data: Binary data (bytes)
        var_name: Variable name for the C array
    
    Returns:
        C header file content as string
    """
    c_str = ''
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
    c_str += 'unsigned char ' + var_name + '[] = {'

    hex_array = []
    for i, val in enumerate(hex_data):
        hex_str = format(val, '#04x')

        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '

        hex_array.append(hex_str)

    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str

def convert_tflite_to_c(tflite_path='tflite_actors/model.tflite', output_path='model.h', var_name='model'):
    """
    Convert TFLite model to C byte array header file.
    
    Args:
        tflite_path: Path to TFLite model file
        output_path: Path to save C header file
        var_name: Variable name for the C array
    """
    print("=" * 70)
    print("Converting TFLite to C Byte Array (Cell 15 from examples.ipynb)")
    print("=" * 70)
    
    # Check if TFLite file exists
    print(f"\n1. Checking TFLite file...")
    if not os.path.exists(tflite_path):
        print(f"   [X] ERROR: TFLite file not found at {tflite_path}")
        return False
    
    file_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    print(f"   [OK] TFLite file found: {tflite_path}")
    print(f"   File size: {file_size:.4f} MB ({os.path.getsize(tflite_path)} bytes)")
    
    # Read TFLite file as binary
    print("\n2. Reading TFLite file...")
    try:
        with open(tflite_path, 'rb') as f:
            tflite_bytes = f.read()
        print(f"   [OK] Read {len(tflite_bytes)} bytes")
    except Exception as e:
        print(f"   [X] ERROR reading file: {e}")
        return False
    
    # Convert to C array
    print("\n3. Converting to C byte array...")
    try:
        c_header = hex_to_c_array(tflite_bytes, var_name)
        print(f"   [OK] C header generated ({len(c_header)} characters)")
    except Exception as e:
        print(f"   [X] ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Write C header file
    print(f"\n4. Writing C header file...")
    try:
        with open(output_path, 'w') as f:
            f.write(c_header)
        print(f"   [OK] C header file saved: {output_path}")
    except Exception as e:
        print(f"   [X] ERROR writing file: {e}")
        return False
    
    # Verify file
    print("\n5. Verifying C header file...")
    if os.path.exists(output_path):
        header_size = os.path.getsize(output_path) / 1024  # KB
        print(f"   [OK] C header file exists")
        print(f"   File size: {header_size:.2f} KB")
        print(f"   Location: {output_path}")
        
        # Show first few lines
        with open(output_path, 'r') as f:
            lines = f.readlines()[:10]
            print(f"\n   First 10 lines:")
            for i, line in enumerate(lines, 1):
                print(f"   {i:2d}: {line.rstrip()}")
    else:
        print(f"   [X] ERROR: C header file not found")
        return False
    
    print("\n" + "=" * 70)
    print("C BYTE ARRAY CONVERSION COMPLETE [OK]")
    print("=" * 70)
    print(f"\nC header file saved to: {output_path}")
    print(f"\nThe model is now ready for ESP32 deployment!")
    print(f"\nTo use in ESP32 code:")
    print(f"  #include \"{output_path}\"")
    print(f"  // Then use: {var_name} and {var_name}_len")
    
    return True

if __name__ == '__main__':
    success = convert_tflite_to_c()
    exit(0 if success else 1)

