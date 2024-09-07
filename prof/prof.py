import os
import time
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# config
# ===============================

args = {}
args['seed'] = 1234
args['main'] = Path(__file__)
args['cwd'] = args['main'].parent.parent
args['src'] = args['cwd'] / 'src'
args['hpp'] = args['src'] / 'headers.hpp'
args['cmdstan'] = Path(os.path.expanduser("~")) / '.cmdstan' / 'cmdstan-2.35.0'
args['stanc_args'] = {"include-paths": [str(args['src'])]}
args['cpp_options'] = {"LDFLAGS": f"{str(args['src'])}/bnb/bnb_lpmf.o -L{os.environ['CUDA_HOME']}/lib64 -lcudart"}
args['stan_dir'] = args['cwd'] / 'model'

import sys
sys.path.append(str(args['cwd']))
from utils import stan_model, beta_neg_binomial_rng


# ===============================
# execution
# ===============================

command = ["nvcc", "-O3", "-lineinfo", "-c", str(args['src'])+"/bnb/bnb_lpmf.cu", "-o", str(args['src'])+"/bnb/bnb_lpmf.o"]
result = subprocess.run(command, capture_output=True, text=True)
print("stdout:", result.stdout)
print("stderr:", result.stderr)

data_sizes = [4000, 6000, 8000, 10000, 12000, 14000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000]
results = []

def run_and_measure_time(model, data_size):
    data = beta_neg_binomial_rng(r=6, alpha=4, beta=0.5, y_max=500, size=data_size, seed=args['seed'])
    stan_data = {
        'N': len(data),
        'y': data,
    }
    start_time = time.time()
    model.sample(stan_data, 
                 **{"chains": 4, "iter_warmup": 500, "iter_sampling": 500, 
                    "show_console": False, "seed": args['seed'], "refresh": 20})
    elapsed_time = time.time() - start_time
    return elapsed_time

cpu_model = stan_model(args['stan_dir'] / 'model_bnb.stan')
cpu_model.compile(user_header=args['hpp'],
                stanc_options=args['stanc_args'])
gpu_model = stan_model(args['stan_dir'] / 'model_cu_bnb.stan')
gpu_model.compile(user_header=args['hpp'],
                stanc_options=args['stanc_args'],
                cpp_options=args['cpp_options'])

for size in data_sizes:
    # Test CPU version
    cpu_time = run_and_measure_time(cpu_model, size)
    results.append({'Data Size': size, 'Model Type': 'CPU', 'Time (seconds)': cpu_time})
    # Test GPU version
    gpu_time = run_and_measure_time(gpu_model, size)
    results.append({'Data Size': size, 'Model Type': 'GPU', 'Time (seconds)': gpu_time})

df = pd.DataFrame(results)
df.to_csv(args['cwd'] / 'output' / 'model_performance.csv', index=False)



# ===============================
# comparison
# ===============================

df_cpu = df[df['Model Type'] == 'CPU'].set_index('Data Size')
df_gpu = df[df['Model Type'] == 'GPU'].set_index('Data Size')
df_ratio = (df_cpu['Time (seconds)'] / df_gpu['Time (seconds)']).reset_index()
df_ratio.columns = ['Data Size', 'CPU/GPU Ratio']

fig = make_subplots(rows=1, cols=2, 
                    horizontal_spacing=0.05,
                    vertical_spacing=0.03,
                    subplot_titles=('Time (seconds)', 'CPU/GPU Time Ratio')
                    )

fig.add_trace(
    go.Scatter(x=df[df['Model Type'] == 'CPU']['Data Size'], y=df[df['Model Type'] == 'CPU']['Time (seconds)'], mode='lines', name='CPU'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df[df['Model Type'] == 'CPU']['Data Size'], y=df[df['Model Type'] == 'GPU']['Time (seconds)'], mode='lines', name='GPU'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_ratio['Data Size'], y=df_ratio['CPU/GPU Ratio'], mode='lines', name='CPU/GPU'),
    row=1, col=2
)

# fig.update_yaxes(title_text='Time (seconds)', row=1, col=1)
# fig.update_yaxes(title_text='CPU/GPU Ratio', row=1, col=2)
fig.update_xaxes(title_text='Data Size', row=1, col=1)
fig.update_xaxes(title_text='Data Size', row=1, col=2)

fig.update_yaxes(zeroline=False,)
fig.update_xaxes(dtick="20000")
fig.update_layout(title_text='', 
                  showlegend=True,
                  height=500,
                  width=1200)
fig.write_image(args['cwd'] / 'output' / 'model_performance.pdf')



