DEVICE_OPTIONS = {  
    "910B3": {  
        "NAME": "910B3",  
        "CP": 280,  # T 算力  
        "BW": 800,  # GB/s 带宽  
        "MC": 32,  # GB 内存  
        "MFU": 0.8,  # 算力利用率  
        "MMU": 0.8,  # 内存利用率  
        "MBU": 0.8,  # 带宽利用率  
        "PMFU": 0.8,  # 首 token 算力利用率  
        "core": -1,  # 核心数  
        "frequency": -1,  # 频率  
        "description": "910B3，280T算力，主要应用于推理场景，"  
    },  
    # 可以添加更多设备  
    "910B2": {  
        "NAME": "910B2",  
        "CP": 376,  
        "BW": 800,  
        "MC": 64,  
        "MFU": 0.83,  
        "MMU": 0.8,  
        "MBU": 0.85,  
        "PMFU": 0.8,  
        "core": -1,  
        "frequency": -1,  
        "description": "910B2，内存空间大，适用于训练场景"  
    }  
}  


FP_MAP = {
    "FP16":2,
    "FP32":4,
    "INT8":1,
    "INT4":0.5,
    "BP16":2
}