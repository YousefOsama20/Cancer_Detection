import h5py
import json

def patch_h5(filepath):
    print("Patching:", filepath)
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' not in f.attrs:
                print("No model_config found")
                return
            s = f.attrs['model_config']
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            conf = json.loads(s)
            
            def scrub(obj):
                if isinstance(obj, dict):
                    # Remove bad keys for Keras 3
                    obj.pop('quantization_config', None)
                    
                    # Fix input layer batch_shape bug in Keras 3
                    if obj.get('class_name') == 'InputLayer' and 'config' in obj:
                        c = obj['config']
                        if 'batch_shape' in c and 'batch_input_shape' not in c:
                            c['batch_input_shape'] = c.pop('batch_shape')
                        # 'optional' is not recognized in Keras 3 InputLayer
                        c.pop('optional', None)
                        
                    for k in list(obj.keys()):
                        scrub(obj[k])
                elif isinstance(obj, list):
                    for item in obj:
                        scrub(item)
            
            scrub(conf)
            f.attrs['model_config'] = json.dumps(conf).encode('utf-8')
            print("Successfully patched!")
    except Exception as e:
        print("Failed:", e)

# Patch both models
patch_h5(r'Brain Tumor_EfficientNetB0\Brain Tumor_EfficientNetB0.h5')
patch_h5(r'Breast Cancer U_Net\Breast Cancer U_Net.h5')
