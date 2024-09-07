import os
import subprocess
import numpy as np
import pandas as pd
import arviz as az
import plotly.express as px
from pathlib import Path
from datetime import datetime
from cmdstanpy import write_stan_json

# ===============================
# config
# ===============================

args = {}
args['seed'] = 1234
args['main'] = Path(__file__)
args['cwd'] = args['main'].parent
args['src'] = args['cwd'] / 'src'
args['hpp'] = args['src'] / 'headers.hpp'
args['cmdstan'] = Path(os.path.expanduser("~")) / '.cmdstan' / 'cmdstan-2.35.0'
args['stan_dir'] = args['cwd'] / 'model'
args['stanc_args'] = {"include-paths": [str(args['src'])]}
args['cpp_options'] = {"LDFLAGS": f"{str(args['src'])}/bnb/bnb_lpmf.o -L{os.environ['CUDA_HOME']}/lib64 -lcudart"}

import sys
sys.path.append(str(args['cwd']))
from utils import stan_model, beta_neg_binomial_rng

data = beta_neg_binomial_rng(r=6, alpha=4, beta=0.5, y_max=500, size=10000, seed=args['seed'])
stan_data_0 = {
    'N': len(data),
    'y': data,
}
write_stan_json(args['cwd'] / 'output/data.json', stan_data_0)

# ===============================
# execution
# ===============================

command = ["nvcc", "-O3", "-lineinfo", "-c", str(args['src'])+"/bnb/bnb_lpmf.cu", "-o", str(args['src'])+"/bnb/bnb_lpmf.o"]
result = subprocess.run(command, capture_output=True, text=True)
print("stdout:", result.stdout)
print("stderr:", result.stderr)


model = stan_model(args['stan_dir'] / 'model_bnb.stan')
model.compile(user_header=args['hpp'],
              stanc_options=args['stanc_args'],
            #   cpp_options=args['cpp_options']
              )
model.sample(stan_data_0, 
            **{"chains": 4, "iter_warmup": 1000, "iter_sampling": 1000, 
            "show_console": False, "seed": args['seed'], "refresh": 20,})
model.diagnose()
model.save()


model_cu = stan_model(args['stan_dir'] / 'model_cu_bnb.stan')
model_cu.compile(user_header=args['hpp'],
              stanc_options=args['stanc_args'],
              cpp_options=args['cpp_options'])
model_cu.sample(stan_data_0, 
            **{"chains": 4, "iter_warmup": 1000, "iter_sampling": 1000, 
            "show_console": False, "seed": args['seed'], "refresh": 20,})
model_cu.diagnose()
model_cu.save()

# ===============================
# comparison
# ===============================
params = ['r', 'a', 'b']
df_list = []

for param in params:
    param_cpu = model.fit.stan_variable(param).flatten()
    param_gpu = model_cu.fit.stan_variable(param).flatten()

    df_param = pd.DataFrame({
        'value': list(param_cpu) + list(param_gpu),
        'type': ['CPU'] * len(param_cpu) + ['GPU'] * len(param_gpu),
        'parameter': [param] * (len(param_cpu) + len(param_gpu))
    })
    
    df_list.append(df_param)

df_all_params = pd.concat(df_list, ignore_index=True)

fig = px.histogram(df_all_params, x='value', color='type', facet_col='parameter', 
                   barmode='group', opacity=0.7, 
                   labels={'value': 'Posterior Value', 'type': 'Model Type', 'parameter': 'Parameter'},
                   title='Comparison of Posterior Distributions for CPU and GPU Models')

fig.write_image(args['cwd'] / 'output/posterior_comparison.pdf')


