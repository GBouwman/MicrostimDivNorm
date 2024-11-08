import Adapted_DN_Model
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg') # TkAgg
import matplotlib.pyplot as plt

# Model parameters
sample_rate = 1000
shift = 0.654
scale = 4.33
tau = 0.005
n = 1.7
sigma = 0.474
sigma_a = 4.63
tau_b = 29.1
tau_a = 0.617
alpha = 1.58
baseline = 0.149

# Set SI values to use
SIs = [0.3,0.5,0.7,1,2,3]
# Set time to simulate
timelim = 20 # seconds

# Loop and generate
fig,ax = plt.subplots(len(SIs),sharex=True,sharey=True)
for i,SI in enumerate(SIs):
    #nr reps
    repsplus5 = int((timelim / SI) + 5)
    stim = np.tile(np.concatenate((np.repeat([1],167),np.repeat([0],(SI*1000)-167))),repsplus5)
    stim = stim[:int(timelim*1000)]

    model = Adapted_DN_Model(stim,sample_rate,shift,scale,tau,n,sigma,tau_a,tau_b,alpha,baseline,sigma_a=sigma_a)

    stim_shift = model.response_shift(stim)
    linear = model.lin(stim_shift)
    linear_rectf = model.rectf(linear)
    linear_rectf_exp = model.exp(linear_rectf)
    rsp, demrsp, demrsp_short = model.norm_delay(linear_rectf_exp, linear)

    ax[i].plot(np.linspace(0,timelim,len(stim)),rsp,label=f'{SI}s')
    ax[i].legend(loc='upper right')
ax[-1].set_xlabel('Time [s]')
ax[0].set_title('Predicted neural response for various values of SI')
fig.show()
