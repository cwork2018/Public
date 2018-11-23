import time
import numpy as np
import cupy as cp

try:
  import cupy.cuda.device
  print("cupy import [ok]")
except ImportError:
  print("cupy import [error]")

nloopa = 5
nloopb = 300

def f_cal(ff, ii,mi,mc,mj):
  maa = ff.arange(mi*mc).reshape(mi,mc) / float(mi*mc) + (float(ii) * 0.1)
  #mbb = 0.001 + ff.sin(ff.arange(mc*mj).reshape(mc,mj) * 0.1)
  mbb = ff.sin(ff.arange(mc*mj).reshape(mc,mj) * 0.1)

  tstart = time.time()
  #-------------- start --------------
  #mpp = maa.reshape(-1,mc)
  #mqq = mbb.reshape(mc,-1)
  #mrlta = ff.dot(mpp, mqq); mrlta = mrlta.reshape(mi,mj)
  #asum = ff.sum(mrlta)
  for jj in range(nloopb):
    mpp = maa.reshape(-1,mc)
    mqq = mbb.reshape(mc,-1)
    mrlta = ff.dot(mpp, mqq)
    maa = mrlta.reshape(mi,mj)
  asum = ff.sum(mrlta)
  #-------------- end --------------
  tend = time.time()
  return tend-tstart, asum

nn = 1000
mdd = np.dot(np.arange(nn*nn).reshape(nn,nn), np.arange(nn*nn).reshape(nn,nn))
mdd = cp.dot(cp.arange(nn*nn).reshape(nn,nn), cp.arange(nn*nn).reshape(nn,nn))

#ni,nc,nj = 5000, 10000, 1000
ni,nc,nj = 1000, 1000, 1000

for iloop in range(nloopa):
  #mi,mc,mj = ni,nc,nj
  kk = 100
  mi,mc,mj = ni+iloop*kk, nc+iloop*kk, nj+iloop*kk

  dtb, sumb = f_cal(cp, iloop,mi,mc,mj)
  dta, suma = f_cal(np, iloop,mi,mc,mj)
  ratio = min(10000, dta / (dtb + 1.0e-30))
  err = suma-sumb

  print("iloop=[%2d] [%d,%d]x[%d,%d]x%d dtime_{numpy,cupy}=[%12.9g %12.9g] r=[%7.1f] sum=[%.7g %.7g %.7g]" \
    % (iloop, mi,mc,mc,mj,nloopb, dta,dtb, ratio, suma,sumb,err))
