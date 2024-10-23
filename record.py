import os


#Q[NUM]_[VAR]  NUM 是用于存储权重的位数
GGUF_MAP = {
    "Q8_0": 1, #几乎就等于FP16
    "Q6_K":1.005, # Q8_0 用于全部的张量，稍微比fp16大一点点
    "Q5_K_S":  1#Q5_k用于所有的张量
}

FP_MAP = {
    "FP16":2,
    "FP32":4,
    "INT8":1,
    "INT4":0.5,
    "BP16":2
}

DEVICE_PARAM={
    "NAME" :"910B3",
    "CP" : 280, #T 算力
    "BW" : 800, #GB/s 带宽
    "MC" : 32, #GB 内存
    "MFU" : 0.8, #算力利用率
    "MMU" : 0.8, #内存利用率
    "MBU" : 0.8, #带宽利用率
    "PMFU" : 0.8, #首token算力利用率

    "core": -1, # 核心数
    "frequency": -1, # 频率
    "description":"910B3，280T算力，主要应用于推理场景，",
}

def init_attention_list(d, attention_dim_list):
    if attention_dim_list[0] == -1:
        d_q = d
    else:
        d_q = attention_dim_list[0]
        
    
    if attention_dim_list[1] == -1:
        d_k = d
    else:
        d_k = attention_dim_list[1]
        
    
    if attention_dim_list[2] == -1:
        d_v = d
    else:
        d_v = attention_dim_list[2]
        
    
    if attention_dim_list[3] == -1:
        d_o = d
    else:
        d_o = attention_dim_list[3]

    return d_q,d_k,d_v,d_o



def tf_param_count(l, d, d_f=-1,h_q=1, h_k=-1, h_v=-1, attention_dim_list=[-1,-1,-1,-1], v_dim = 0,EASY_MODE=True):
    '''
    l: decoder_layer_num
    d:attention_hidden_dim
    d_f:feedforward_dim : -1 means d*4  
    feedforward = (linear1 + relu + linear2 + normalization)
    
    attention_dim_list: [d_q,d_k, d_v, d_o]
    
    head_count_list = [h_q,h_k,h_v]
    
    v_dim: embedding 词表的大小，为0的时候默认不计算embedding模型参数量
    EASY_MODE: True 表示忽略两级较少的干扰项，False表示精确计算
    
    https://blog.csdn.net/beingstrong/article/details/132383758
    https://zhuanlan.zhihu.com/p/677774901
    
    12*l*d*d
    '''
    
    if d_f == -1:
        d_f = d*4
        
    d_q,d_k,d_v,d_o = init_attention_list(d,attention_dim_list)
    
    if h_k == -1:
        h_k = h_q
    if h_v == -1:
        h_v = h_q
    if EASY_MODE:
        self_attention_bias_p = 0
        freeforward_bias_p = 0
    else:
        self_attention_bias_p = d_q + d_k + d_v + d_o
        freeforward_bias_p = d_f + d  # w[d,df] + w[df,d] 两个bias偏置一个和df相同，一个和d相同
        
    
    embedding_p = v_dim*d #一般默认
    self_attention_p = d_q*d_q + d_k*d_k + d_v*d_v + d_o*d_o #W_Q,W_K,W_V,W_O 四个矩阵的参数
    freeforward_p = d_f*d +  d_f*d
    transformer_c = l*(self_attention_p + freeforward_p + self_attention_bias_p + freeforward_bias_p) + embedding_p
    
    return transformer_c
    
def tf_calculate_count(l, d, b,s,d_f=-1, h=1, attention_dim_list=[-1,-1,-1,-1],EASY_MODE=True):
    '''
    C: 总计算量 flops
    P: 模型总参数量
    D: token总数量
    有一个近似公式: C = 6*P*D
    反向传播可以认为激活和权重求梯度，有两次矩阵乘，前向只有一次，所以
    反向传播的计算量是前向的两倍
    https://zhuanlan.zhihu.com/p/672604715
    l: decoder_layer_num
    d:attention_hidden_dim
    d_f:feedforward_dim : -1 means d*4
    b: batch_size
    s: sequence_length
    attention_dim_list: [d_q, d_k, d_v, d_o]
    X_e: 输入X经过embedding后的结果，维度为d
    一个mn，和一个np的矩阵计算，都需要进行2nmp次运算（+和*）
    EASY_MODE: True 表示忽略两级较少的干扰项，False表示精确计算
    https://zhuanlan.zhihu.com/p/672604715 (训练的计算量是推理的2倍)
    
    2*p*c
    '''
    if d_f == -1:
        d_f = d*4
        
    d_q,d_k,d_v,d_o = init_attention_list(d,attention_dim_list)
    
    
    if EASY_MODE:
        self_attention_bias_p = 0
        freeforward_bias_p = 0
        soft_max_p = 0

    else:
        self_attention_bias_p = b * s * (d_q + d_k + d_v + d_o)
        freeforward_bias_p = b * s * (d_f + d)
        soft_max_p = h * b * s * s  #[s*s的矩阵]
        
    # (X_e * W_q) = [s,d] * [d,d] 多头除以h 再拼接h个头，结果不变
    C_0 = b * (s*d*d_q*2 + s*d*d_k*2 + s*d*d_v*2)
    # q*k + score * v  [s,d] * [d*s] + [s,s] * [s,d] #多头除以n，再拼接，结果依然不变
    C_1 = b* ((2*s*d*s) + (2*s*s*d))
    # z*wo = [s,d] * [d,d]
    C2 = b * (2*s*d*d_o) 
    
    
    #forward 升维度 [s,d],[d,4d] 
    C_up = b * 2 * s * d * d_f
    #forward 降维度 [s,4d],[4d,d]
    C_down = b * 2 * s * d_f * d
    
    C_infer = l* (C_0 + C_1 + C2 + C_up + C_down + self_attention_bias_p + freeforward_bias_p + soft_max_p )
    
    C_training = C_infer * 3 #单个block的
    return C_infer, C_training
    


def kv_memori_size(s,l,d,b,q_head=1, kv_head=1,group=1, pb='FP16',kvb='FP16',KV_METHOD='DEFAULT'):
    '''
    s: sequence
    l: layer
    d:token_dim
    b:batch_size
    KV_METHOD: 
        DEFAULT - 默认所有kv都存的
        MQA - 多Q共用一个KVCACHE
        GQA - 一个Q对应一个KVCACHE
    https://zhuanlan.zhihu.com/p/685853516
    '''
    # KV缓存各一个，所以×2 根据kvchache的公式，可以看到每一个都保存好最开始计算的 s*d(token embedding维度的结果)
    if KV_METHOD == 'DEFAULT':
        kvcache_size = 2 *  FP_MAP[kvb] * s * l * d * b
    elif KV_METHOD == 'MQA':
        kvcache_size = 2 *  FP_MAP[kvb] * s * l * d * b * kv_head/q_head #(每一组都合并了n个kv_cache,所以少了n倍)
    else:
        kvcache_size = 2 *  FP_MAP[kvb] * s * l * d * b * group * kv_head/q_head
    
    #返回GB大小的内存
    return kvcache_size/1024/1024/1024

def kv_calculate(l,s,d):
    '''
    l: layer
    s: sequence
    d: token_dim
    '''
    #每一个kv的计算量 [s,d] * [d,d] = 2*s*d*d 这是完整的self attention 计算，目前每次只要计算s*d, d*1 的量
    kv_calculate = 2 * s * d * l
    return kv_calculate

def evaluate_memory(p_memory_size, kv_memory_size, tp_count, device_param):
    '''
    l: layer
    d: token_dim
    b: batch_size
    s: sequence
    attention_dim_list: [d_q,d_k,d_v,d_o]
    MMU: 容量利用率，一般不会等于1，这里默认为0.8
    device_param : 设备的各种不同的参数
    '''
    #计算模型参数量

    kv_memory_rate = (device_param['MC']*device_param['MMU'] - p_memory_size/tp_count)/kv_memory_size
    if kv_memory_rate < 0:
        return 0,"模型内存不够"
    elif kv_memory_rate >= 1:
        return kv_memory_rate,"足够装载KV CACHE"
    else:
        return kv_memory_rate,"KV CACHE内存不够"


def evaluate_bandwidth(p_memory_size, kv_memory_size, user_tps, tp_count, device_param,):
    '''
    p_memory_size: 模型参数量
    user_tps: 用户的tps
    tp_count: 多张卡张量并行均分，用于计算每张卡的传输参数量
    '''
    
    bw_rate = (device_param['BW']*device_param['MBU'] - user_tps*p_memory_size/tp_count) // (kv_memory_size*user_tps)
    if bw_rate < 0:
        return 0,"带宽不够"
    elif bw_rate >= 1:
        return bw_rate,"足够带宽"
    else:
        return bw_rate,"kv cache的带宽不够"

def evaluate_count(paramter, kv_calculate, user_tps,device_param):
    '''
    不论是不是分布式的，为了保证1s内推理完成，n张卡，那么模型需要计算的时间耗时就应该是完整的1/n
    所以最后单卡等于需要能完整的计算完整模型推理一个的量。
    这种公式归纳，预估的计算量，实际上比需要的计算量更小一些，慎重！
    todo: kv cahe 也×2的推导
    由公式2param * D = 计算量可知
    kvm = kvc * kvb * head_rate = kv_p * kvb * tp_count
    
    '''

    
    cp_rate = (device_param['CP']*device_param['MFU'])/((user_tps*(kv_calculate+paramter)*2)/1024/1024/1024/1024)
    if cp_rate < 1:
        return 0,"计算量不够"
    elif cp_rate >= 1:
        return cp_rate,"足够算力"

def evaluate_all(input_len,output_len,l,d,b,device_param,user_tps,user_count,tp_count,q_head=1,kv_head=1,group=1,v_dim=30000,EASY_MODE=False,attention_dim_list=[-1,-1,-1,-1],pb='FP16',kvb='FP16',KV_METHOD='MQA',D_COUNT=8):
    '''
    TTFT:全推理延迟： 首TOKEN推理延迟 + 首缓存推理延迟
    D_COUNT: 每台机器的卡数
    
    S_NUM 计算可以认为是分成两个部分。 第一部分机器用于处理正常推理， user_count/SUP, 需要这么多张卡来支持用户的推理请求
    第二部分是 首次推理时候 延误了ttft 秒，每秒都需要产生user_tps 这么多的token， user_count这么多用户一共产生了 大量的token, 同时，由于是串行推理
    每个卡都只能占用 1/tp_count 这么多时间，实际的算力是需要 翻tp_count倍的，总共的token数/output。 为了满足用户访问的QPS，这么多
    '''
    s = input_len + output_len
    paramter = tf_param_count(l,d,v_dim=v_dim,EASY_MODE=EASY_MODE,attention_dim_list=attention_dim_list)
    p_memory_size = paramter * FP_MAP[pb]/1024/1024/1024
    print(f"模型参数量: {paramter/1e9:.2f}（B），占用显存：{p_memory_size:.2f}（GB）")
    kv_memory_size = kv_memori_size(s,l,d,b,q_head,kv_head,group,pb,kvb,KV_METHOD=KV_METHOD)
    kv_calculate_count = kv_calculate(l,s,d)
    # paramter_unit = paramter/1024/1024/1024/1024
    # kv_calculate_unit = kv_calculate_count/1024/1024/1024/1024
    cp_rate,cp_res = evaluate_count(paramter,kv_calculate_count, user_tps,device_param)
    bw_rate,bw_res = evaluate_bandwidth(p_memory_size, kv_memory_size, user_tps, tp_count, device_param)
    kv_rate,kv_res = evaluate_memory(p_memory_size, kv_memory_size, tp_count, device_param)
    SUP = min(cp_rate,bw_rate,kv_rate) #单卡最多支持的用户并发数
    
    # 2p*seqlen 是模型本身需要的推力量， 2*kvc*input_len 是kv_caceh需要的推力量。因为这个需要经过8张卡的加工，是并行的，所以需要*tp_count
    ttft = (2*paramter + 2*input_len*l*d)*input_len/(device_param['CP']*device_param['PMFU']*tp_count)/1024/1024/1024/1024
    
    # 支持这么多用户需要的卡数 + 在规定的延迟内，完成outputlen这么长的推理，需要的卡数
    S_num = (user_count/SUP + user_count*ttft*tp_count*user_tps/output_len)/D_COUNT
    print(f"单卡最多支持的用户并发数：{SUP} (CPR:{cp_rate}, BWR:{bw_rate}, KVR:{kv_rate}). 需要服务器数量:{S_num} (TTFT:{ttft})")
    return S_num * D_COUNT, (paramter,p_memory_size,kv_memory_size, cp_rate,bw_rate,kv_rate,SUP,ttft)

def test1():
    layer=90
    token_dim=8192
    batch_size = 2
    sequence = 4096
    head_count=64
    paramter = tf_param_count(layer,token_dim,v_dim=30000,EASY_MODE=False)
    print(f"参数量估计: {paramter/1e9:.2f}（B）")
    print(f"公式预测参数量: {12*layer*token_dim*token_dim/1e9:.2f}（B）")
    
    
    C_infer, C_training = tf_calculate_count(layer,token_dim,batch_size,sequence,h=head_count, EASY_MODE=False)
    print(f"推理计算量: {C_infer/1024/1024/1024/1024:.2f}（T）")
    print(f"训练计算量: {C_training/1024/1024/1024/1024:.2f}（T）")
    print(f"公式预测推理计算量: {2*(paramter)*sequence*batch_size/1024/1024/1024/1024:.2f}（T）")
    print(f"公式预测训练计算量: {6*paramter*sequence*batch_size/1024/1024/1024/1024:.2f}（T）")
    

def test2(device_param):
    layer=90
    token_dim=8192
    batch_size = 1
    head_count=64
    kv_count = 8 
    user_count = 1000000
    user_tps = 20
    input_len = 2048
    output_len = 2048
    tp_count = 8
    server_gpu_count = 8
    PB='FP16'
    KVB='FP16'
    gpu_count = evaluate_all(input_len,output_len,layer,token_dim,batch_size,device_param,user_tps,user_count,tp_count,q_head=head_count,kv_head=kv_count,group=1,v_dim=30000,EASY_MODE=False,attention_dim_list=[-1,-1,-1,-1],pb=PB,kvb=KVB,KV_METHOD='MQA',D_COUNT=server_gpu_count)
    return gpu_count
if __name__ == "__main__":
    device_param =  DEVICE_PARAM
    test1()
    test2(device_param)
