import time
import numpy as np
import cupy as cp

try:
  import cupy.cuda.device
  print("cupy import [ok]")
  bcupy = True
except ImportError:
  print("cupy import [error]")
  bcupy = False

#bcupy = False
#print("bcupy=[%s]" % bcupy)

nloopa = 5
nloopb = 30

def f_cal(ff, ii,mc):
  maa = ff.arange(mc*mc).reshape(mc,mc) / float(mc*mc) + (float(ii) * 0.1)
  mbb = ff.sin(ff.arange(mc*mc).reshape(mc,mc) * 0.1)
  #maa = ff.zeros((mc,mc)) + 0.0
  #mbb = ff.zeros((mc,mc)) + 0.0

  tstart = time.time()
  #-------------- start --------------
  for jj in range(nloopb): maa = ff.dot(maa, mbb)
  #-------------- end --------------
  tend = time.time()
  asum = ff.sum(maa)
  return tend-tstart, asum

#nc = 5000
nc = 2000

for iloop in range(nloopa):
  kk = 100
  mc = nc+iloop*kk

  nsize_gb = 2*mc*mc*8 * 1e-9

  if bcupy:
    dta, suma = f_cal(np, iloop,mc)
    dtb, sumb = f_cal(cp, iloop,mc)
    ratio = min(10000, dta / (dtb + 1.0e-30))
    err = suma-sumb
  
    print("iloop=[%2d] [%d,%d]x[%d,%d]x%d mem= %7.3fgb dtime_{numpy,cupy}=[%7.3f %9.6f] r=[%7.1f] sum=[%.7g %.7g %.7g]" \
      % (iloop, mc,mc,mc,mc,nloopb, nsize_gb, dta,dtb, ratio, suma,sumb,err))

  else:
    dta, suma = f_cal(np, iloop,mc)
    print("iloop=[%2d] [%d,%d]x[%d,%d]x%d mem= %7.3fgb dtime_{numpy}=[%7.3f] sum=[%.7g]" \
      % (iloop, mc,mc,mc,mc,nloopb, nsize_gb, dta, suma))
