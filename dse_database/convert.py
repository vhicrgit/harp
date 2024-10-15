import pandas as pd
import json

df = pd.read_csv("/root/autodl-tmp/kaggle/dse_database/test.csv")

DESIGNS = {}

for index, row in df.iterrows():
    design = row['designs']
    items = design.split('.')
    point = {}
    kernel = None
    for item in items:
        if '__kernel__' in item:
            key = '__kernel__'
            value = item.replace('__kernel__-', '')
            if value == 'stencil':
                value = 'stencil_stencil2d'
            kernel = value
            
        elif '__version__' in item:
            continue
        else:
            key, value = item.split('-')
            if value.isdigit():
                value = int(value)
            if value == 'NA':
                value = ''
            point[key] = value
    
    if kernel not in DESIGNS:
        DESIGNS[kernel] = {}
        
    DESIGNS[kernel][design] = {
        'id': index,
        'point': point,
        'kernel_name': kernel,
        'design_name': design
    }
    
    
with open('test.json', 'w') as f:
    json.dump(DESIGNS, f)
        
    