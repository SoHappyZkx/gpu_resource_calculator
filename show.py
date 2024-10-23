import dash  
from dash import dcc, html  
from dash.dependencies import Input, Output  
from record import evaluate_all,FP_MAP  
from device_config import DEVICE_OPTIONS  
import math

# 初始化Dash应用  
app = dash.Dash(__name__)  

# 定义应用布局  
app.layout = html.Div([  
      html.Div([  
        html.Img(src='./assets/logo.png', style={'height': '50px', 'margin-right': '20px'}),  
        html.H1("北京公司国产算力计算器", style={'text-align': 'center', 'flex-grow': '1'})   
    ], style={  
        'display': 'flex',  
        'align-items': 'center',  
        'justify-content': 'center',  
        'margin-bottom': '20px'  
    }),  
    
    html.H2("请选择设备型号:"),  
    dcc.Dropdown(  
        id='device-dropdown',  
        options=[{'label': name, 'value': name} for name in DEVICE_OPTIONS.keys()],  
        value='910B3'  # 默认选项  
    ),  
    html.Div(id='device-info'),  # 用于显示设备信息  
    
    html.Div([  
        html.H2("模型参数设置:", style={'margin-right': '20px'}),  
        html.Div([  
            html.Span("模型参数数量(B): ", style={'font-weight': 'bold'}),  
            html.Span(id='model-paramter-count-value'),  
            html.Span(" | 模型内存大小(G): ", style={'font-weight': 'bold', 'margin-left': '10px'}),  
            html.Span(id='model-memorize-size-value')  
        ], style={'display': 'flex', 'align-items': 'center', 'margin-left': '10px'})  
    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),
     
    html.Div([  
            html.Label("模型精度:"),  
            dcc.Dropdown(  
                id='model-precision-dropdown',  
                options=[{'label': precision, 'value': precision} for precision in FP_MAP.keys()],  
                value='FP16',  # 默认选项  
                style={'width': '200px'}  
            )  
        ], style={'display': 'inline-block', 'margin-right': '20px'}),  
  
        # 缓存精度选择  
        html.Div([  
            html.Label("缓存精度:"),  
            dcc.Dropdown(  
                id='cache-precision-dropdown',  
                options=[{'label': precision, 'value': precision} for precision in FP_MAP.keys()],  
                value='FP32',  # 默认选项  
                style={'width': '200px'}  
            )  
        ], style={'display': 'inline-block'}),
     
    html.Div([  
        # 模型层数  
        html.Div([  
            html.Label(["模型层数 (Layer) : ", html.Span(id='layer-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='layer-slider', min=20, max=200, value=90, marks={i: str(i) for i in range(20, 201, 20)}, step=1)  
        ]),  
        
        # 嵌入层维度  
        html.Div([  
            html.Label(["嵌入层维度 (Token Dim) : ", html.Span(id='token-dim-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='token-dim-slider', min=1024, max=16384, value=8192, marks={i: str(i) for i in range(1024, 16385, 2048)}, step=128)  
        ]),  
        
        # 批处理量  
        html.Div([  
            html.Label(["批处理量 (Batch Size) : ", html.Span(id='batch-size-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='batch-size-slider', min=1, max=512, value=1, marks={i: str(i) for i in range(1, 513, 50)}, step=1)  
        ]),  
        
        # 注意力头数  
        html.Div([  
            html.Label(["注意力头数 (Head Count) : ", html.Span(id='head-count-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='head-count-slider', min=1, max=128, value=64, marks={i: str(i) for i in range(1, 129, 16)}, step=1)  
        ]),  
    ], style={'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'gap': '20px'}),  
    
    html.Div([  
        html.H2("应用参数设置:", style={'margin-right': '20px'}),  
        html.Div([  
            html.Span("单人缓存内存(G): ", style={'font-weight': 'bold'}),  
            html.Span(id='kvcache-memorize-size-value'),  
            html.Span(" | 计算速率: ", style={'font-weight': 'bold', 'margin-left': '10px'}),  
            html.Span(id='caculate-rate-value'),  
            html.Span(" | 带宽速率: ", style={'font-weight': 'bold', 'margin-left': '10px'}),  
            html.Span(id='bandwidth-rate-value'),  
            html.Span(" | 缓存速率: ", style={'font-weight': 'bold', 'margin-left': '10px'}),  
            html.Span(id='kvcache-rate-value'),  
            html.Span(" | 单卡支撑用户量(人): ", style={'font-weight': 'bold', 'margin-left': '10px'}),  
            html.Span(id='single-user-gpucount-value'),  
            html.Span(" | 首字等待时间(s): ", style={'font-weight': 'bold', 'margin-left': '10px'}),  
            html.Span(id='ttft-value')  
        ], style={'display': 'flex', 'align-items': 'center', 'margin-left': '10px'})  
    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),

    html.Div([  
        # 缓存头数  
        html.Div([  
            html.Label(["缓存头数 (KV Count) : ", html.Span(id='kv-count-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='kv-count-slider', min=1, max=16, value=8, marks={i: str(i) for i in range(1, 17)}, step=1)  
        ]),  
        
        # 用户数量  
        html.Div([  
            html.Label(["用户数量 (User Count) : ", html.Span(id='user-count-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='user-count-slider', min=50, max=10000, value=1000, marks={i: str(i) for i in range(1000, 10001, 2000)}, step=100)  
        ]),  
        
        # 平均用户秒字符数  
        html.Div([  
            html.Label(["平均用户秒字符数 (User TPS) : ", html.Span(id='user-tps-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='user-tps-slider', min=1, max=100, value=20, marks={i: str(i) for i in range(1, 101, 10)}, step=1)  
        ]),  
        
        # 输入长度  
        html.Div([  
            html.Label(["输入长度 (Input Length) : ", html.Span(id='input-len-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='input-len-slider', min=512, max=4096, value=2048, marks={i: str(i) for i in range(512, 4097, 512)}, step=128)  
        ]),  
        
        # 输出长度  
        html.Div([  
            html.Label(["输出长度 (Output Length) : ", html.Span(id='output-len-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='output-len-slider', min=512, max=4096, value=2048, marks={i: str(i) for i in range(512, 4097, 512)}, step=128)  
        ]),  
        
        # 模型分布卡数  
        html.Div([  
            html.Label(["模型分布卡数 (TP Count) : ", html.Span(id='tp-count-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='tp-count-slider', min=1, max=16, value=8, marks={i: str(i) for i in range(1, 17)}, step=1)  
        ]),  
        
        # 服务器卡数  
        html.Div([  
            html.Label(["服务器卡数 (Server GPU Count) : ", html.Span(id='server-gpu-count-value')],style={'font-weight': 'bold', 'font-size': '16px'}),  
            dcc.Slider(id='server-gpu-count-slider', min=1, max=16, value=8, marks={i: str(i) for i in range(1, 17)}, step=1)  
        ]),  
    ], style={'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'gap': '20px'}),  
    
    # 用于存储设备参数  
    dcc.Store(id='device-param-store'),  
    
    # GPU结果显示  
    html.H2("输出结果 : "),  
    html.Div(id='gpu-output', style={'margin-top': 20}),  
])  

@app.callback(  
    [Output('device-info', 'children'), 
    Output('device-param-store', 'data')], 
    Input('device-dropdown', 'value')  
)  
def update_device_info(selected_device):  
    device_param = DEVICE_OPTIONS[selected_device]  
    return html.Div([  
        html.P([  
            html.Span("设备名称: ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['NAME'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),  
        html.P([  
            html.Span("算力 (CP): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(f"{device_param['CP']} T", style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),   
        html.P([  
            html.Span("带宽 (BW): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(f"{device_param['BW']} GB/s", style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),  
        html.P([  
            html.Span("内存 (MC): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(f"{device_param['MC']} GB", style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),  
        html.P([  
            html.Span("算力利用率 (MFU): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['MFU'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),  
        html.P([  
            html.Span("内存利用率 (MMU): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['MMU'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),   
        html.P([  
            html.Span("带宽利用率 (MBU): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['MBU'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),   
        html.P([  
            html.Span("首token算力利用率 (PMFU): ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['PMFU'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),   
        html.P([  
            html.Span("核心数: ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['core'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),   
        html.P([  
            html.Span("频率: ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(f"{device_param['frequency']} GHz", style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ]),   
        html.P([  
            html.Span("描述: ", style={'font-weight': 'bold', 'font-size': '14px'}),  
            html.Span(device_param['description'], style={'border': '1px solid #ccc', 'padding': '1px 4px', 'border-radius': '3px'})  
        ])  
    ], style={  
        'display': 'grid',  
        'grid-template-columns': 'repeat(3, 1fr)',  
        'gap': '5px',  
        'padding': '5px',  
        'border': '1px solid #ccc',  
        'border-radius': '5px',  
        'background-color': '#f9f9f9'  
    }), device_param  

# 定义回调函数  
@app.callback(  
    [Output('gpu-output', 'children'),  
     
     Output('layer-value', 'children'),  
     Output('token-dim-value', 'children'),  
     Output('batch-size-value', 'children'),  
     Output('head-count-value', 'children'),  
     Output('kv-count-value', 'children'),  
     Output('user-count-value', 'children'),  
     Output('user-tps-value', 'children'),  
     Output('input-len-value', 'children'),  
     Output('output-len-value', 'children'),  
     Output('tp-count-value', 'children'),  
     Output('server-gpu-count-value', 'children'),  
     
     Output('model-paramter-count-value', 'children'),
     Output('model-memorize-size-value', 'children'),
     Output('kvcache-memorize-size-value', 'children'),
     Output('caculate-rate-value', 'children'),
     Output('bandwidth-rate-value', 'children'),
     Output('kvcache-rate-value', 'children'),
     Output('single-user-gpucount-value', 'children'),
     Output('ttft-value', 'children')],  
    [Input('layer-slider', 'value'),  
     Input('token-dim-slider', 'value'),  
     Input('batch-size-slider', 'value'),  
     Input('head-count-slider', 'value'),  
     Input('kv-count-slider', 'value'),  
     Input('user-count-slider', 'value'),  
     Input('user-tps-slider', 'value'),  
     Input('input-len-slider', 'value'),  
     Input('output-len-slider', 'value'),  
     Input('tp-count-slider', 'value'),  
     Input('server-gpu-count-slider', 'value'),  
     Input('device-param-store', 'data'),
     Input('model-precision-dropdown', 'value'),
     Input('cache-precision-dropdown', 'value'),
     ]  
)  
def update_gpu_output(layer, token_dim, batch_size, head_count, kv_count, user_count, user_tps, input_len, output_len, tp_count, server_gpu_count, device_param,pb,kvb):  
    '''
        info = (  
        f"模型层数: {layer}\n"  
        f"嵌入维度: {token_dim}\n"  
        f"批处理量: {batch_size}\n"  
        f"注意力头数: {head_count}\n"  
        f"缓存头数: {kv_count}\n"  
        f"用户数量: {user_count}\n"  
        f"平均秒返回字数: {user_tps}\n"  
        f"输入长度: {input_len}\n"  
        f"输出长度: {output_len}\n"  
        f"分布式并行卡数: {tp_count}\n"  
        f"服务器卡数: {server_gpu_count}\n"  
        f"GPU型号: {device_param['NAME']}\n"  
    )  
    KEY_INFO :
        paramter,p_memory_size,kv_memory_size, cp_rate,bw_rate,kv_rate,SUP,ttft
    '''

    
    SUM_COUNT, KEY_INFO = evaluate_all(input_len, output_len, layer, token_dim, batch_size, device_param, user_tps, user_count, tp_count, q_head=head_count, kv_head=kv_count, group=1, v_dim=30000, EASY_MODE=False, attention_dim_list=[-1, -1, -1, -1], pb=pb, kvb=kvb, KV_METHOD='MQA', D_COUNT=server_gpu_count)  
    
    return (  
        f"需要的GPU数量 : {math.ceil(SUM_COUNT)} | 所需服务器台数 : {math.ceil(SUM_COUNT / server_gpu_count)}",  
        f"{layer}", f"{token_dim}", f"{batch_size}", f"{head_count}", f"{kv_count}",   
        f"{user_count}", f"{user_tps}", f"{input_len}", f"{output_len}",   
        f"{tp_count}", f"{server_gpu_count}" ,
        f"{KEY_INFO[0]/1e9:.2f}",f"{KEY_INFO[1]:.2f}",f"{KEY_INFO[2]:.2f}",f"{KEY_INFO[3]:.2f}",f"{KEY_INFO[4]:.2f}",f"{KEY_INFO[5]:.2f}",f"{KEY_INFO[6]:.2f}",f"{KEY_INFO[7]:.2f}"
    )  

# 运行应用  
if __name__ == '__main__':  
    app.run_server(debug=False)