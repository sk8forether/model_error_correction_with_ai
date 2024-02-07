import numpy as np
from check_model import load_linear, compute_skill


def compute_skill_notorch(y, y_pred):
   y_pred_ts=y_pred.reshape((y.shape[0], np.prod(y.shape[1:])))
   y_ts=y.reshape((y.shape[0], np.prod(y.shape[1:])))
   skill = 1-np.mean((y_pred_ts-y_ts)**2,axis=1)/np.mean((y_ts)**2,axis=1)
   return skill

start_day = 0
end_day = 365
length_days=end_day-start_day
varn = 't'

fn_truth = f'npys/y_{varn}.npy'

y = np.load(fn_truth)
ts = np.arange(y.shape[0])/4

y_pred_linear = load_linear(ts, y, start_day, end_day)

skill = compute_skill_notorch(y, y_pred_linear)

np.save(f'npys/skill_linear_{varn}_{end_day}_{length_days}.npy', skill.astype(float))


