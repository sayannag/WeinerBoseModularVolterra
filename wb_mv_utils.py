import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class WeinerBose():
  def __init__(self, isLET=True, isPlot=False):
    self.isLET = isLET
    self.isPlot = isPlot

  def function_generate_laguerre(self, alpha, L, M):
    
    mm = np.arange(0,M-1)
    beta = 1 - alpha
    rootAlpha = np.sqrt(alpha)
    L_buf = np.sqrt((alpha**mm) * beta)

    mmm = np.zeros((M,1))
    Laguerre = np.zeros((len(L_buf),L))
    Laguerre[:,0] = L_buf

    for n1 in range(0,L-1):
        for n2 in mm.tolist():
            mmm[n2,0] = L_buf[n2]
            if n2==0:
              L_buf[n2] = rootAlpha * L_buf[n2]
            else:
              L_buf[n2] = rootAlpha * (L_buf[n2-1] + mmm[n2,0]) - mmm[n2-1,0]
        Laguerre[:,n1+1] = L_buf

    return Laguerre

  def function_LET1_kernelestimate(self, Cest, Laguerre, L, Q):

    M = max(np.shape(Laguerre))

    Kest = {}
    Kest['k0'] = Cest['c0'] 
    k1 = Laguerre @ Cest['c1']
    Kest['k1'] = k1
    r = len(Laguerre[:,0])

    if Q==2: 
        c2 = Cest['c2']
        temp = np.zeros((M, M)) 
        for k1 in range(L):
            for k2 in range(L):
                temp = temp + c2[k1,k2] * Laguerre[:,k1:k1+1] @ Laguerre[:,k2:k2+1].T;   
        Kest['k2'] = temp
        del temp

    return Kest

  def function_Q2self_each_input(self, V):

    N = np.shape(V)[0]
    L = np.shape(V)[1] 

    Nxx = int(L * (L + 1 )/ 2)
    Vxx = np.zeros((N, Nxx))

    cnt = -1
    for k1 in range(L):
        for k2 in range(k1,L):
            cnt = cnt + 1 
            Vxx[:,cnt] = V[:,k1] * V[:,k2]

    return Vxx

  def function_Q1_each_input(self, x, Laguerre):

    N = len(x)
    L = np.shape(Laguerre)[1] 

    Vx = np.zeros((N, L))
    for k in range(L):
        v_column = np.convolve(Laguerre[:,k], x.flatten())   
        Vx[:,k] = v_column[0:N]

    return Vx

  def function_LET1_coeffestimate(self, y, VV, L, Q, Mdiscard):

    y = y.flatten()
    N = len(y) 
    y = y[Mdiscard:N]

    VVorg = VV
    VV = VV[Mdiscard:N,:]

    XtX = VV.T @ VV
    EV, ED = np.linalg.eig(XtX)

    TH_sv = 1e-6 
    if min(abs(np.diag(ED)))>TH_sv:
        ALL_C = np.linalg.inv(XtX) @ VV.T @ y
    else:
        Usvd, Ssvd, Vsvd = np.linalg.svd(VV)
        Vsvd = Vsvd.T
        SV = np.abs(np.diag(Ssvd))
        Idx_sv = int(np.nonzero(SV>=TH_sv)[-1][-1] + 1) # check
        Pseudo_invVV = Vsvd[:,0:Idx_sv] @ np.diag(1/SV[0:Idx_sv]) @ Usvd[:,0:Idx_sv].T
        ALL_C = Pseudo_invVV @ y

    yorg_predict = VVorg @ ALL_C
    y_predict = VV @ ALL_C
    nmse = np.mean((y - y_predict)**2) / np.mean(y**2)

    Cest = {}
    tempC = ALL_C
    Cest['c0'] = tempC[0]
    tempC = tempC[1:]

    c1 = tempC[0:L]
    tempC = tempC[L:]
    Cest['c1'] = c1

    if Q==2:
        c2 = np.zeros((L,L))
        mmm = 0
        for kk in range(L):
            c2[kk,kk:L] = tempC[mmm:mmm+L-kk].T
            mmm = mmm + L - kk # + 1 
        c2 = (c2 + c2.T) / 2
        Cest['c2'] = c2 

    return Cest, yorg_predict, nmse

  def function_LET1_LFparameters(self, x, y, LETparameters):

    alpha = LETparameters['alpha']
    L = LETparameters['L']
    M = LETparameters['M']
    Q = LETparameters['Q']
    Mdiscard = LETparameters['Mdiscard'] 

    x = x.flatten()
    y = y.flatten()
    N = len(y)

    Laguerre = self.function_generate_laguerre(alpha, L, M)

    V0 = np.ones((len(x), 1)) 
    V1 = self.function_Q1_each_input(x, Laguerre) 
    VV_est = np.concatenate((V0, V1),axis=1) 

    if Q==2:
        V2 = self.function_Q2self_each_input(V1)
        VV_est = np.concatenate((VV_est, V2),axis=1)

    Cest, y_predict, nmse = self.function_LET1_coeffestimate(y, VV_est, L, Q, Mdiscard)
    
    Kest = self.function_LET1_kernelestimate(Cest, Laguerre, L, Q)

    return Cest, Kest, y_predict, nmse

  def LET_1(self, x, y, alpha, L, Q, Nfig):

    x = x.flatten()
    y = y.flatten()
    if len(x) != len(y):
        raise NameError('lengths of input and output are different')

    N = len(x)

    if alpha<=0 or alpha>=1:
        print('alpha must be between 0 and 1')
        alpha = 0.5 

    M = (-30 - np.log(1-alpha)) / np.log(alpha)
    M = int(np.ceil(M))

    if L<=0 or L>9: 
        print('L should be between 1 and 9, default vlaue L = 5')
        L = 5

    L = np.round(L)

    if Q!=1 and Q!=2:
        Q = 2 
    
    if Nfig is None:
        Nfig = 1

    Mdiscard = M

    LETparameters = {}
    LETparameters['alpha'] = alpha 
    LETparameters['M'] = M
    LETparameters['Mdiscard'] = Mdiscard
    LETparameters['Q'] = Q
    LETparameters['L'] = L

    Cest, Kest, pred, NMSE = self.function_LET1_LFparameters(x, y, LETparameters)

    if self.isPlot:

      if Nfig is not None: 
          fig, ax1 = plt.subplots(1,1,figsize=(5,3))
          ax1.plot(np.arange(0,M-1), Kest['k1'], linewidth=2) 
          ax1.set_title('LET1: alpha = ' + str(alpha) + ', L = ' + str(int(L)) + ' and Q = ' + str(Q),fontsize=15)
          ax1.grid(True)
          ax1.set_ylabel('k$_1$',fontsize=15) 
          if Q==2:
              xx, yy = np.meshgrid(np.arange(0,M-1),np.arange(0,M-1))
              fig, ax2 = plt.subplots(1,1,figsize=(5,5))
              ax2 = plt.axes(projection='3d')
              ax2.contour3D(xx, yy, np.reshape(np.meshgrid(np.array(Kest['k2'])),(M-1,M-1)), 200)
              ax2.set_zlabel('k$_2$',fontsize=15)
          fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,7))
          ax1.plot(np.arange(0,N), y, 'b',linewidth=2, label='Ground Truth')
          ax1.plot(np.arange(0,N), pred, 'k--',linewidth=2, label='Prediction') 
          ax1.grid(True)
          ax1.set_ylabel('ground truth and model prediction',fontsize=15) 
          ax1.set_title('LET1 analysis: alpha = ' + str(alpha) + ', L = ' + str(int(L)) + ' and Q = ' + str(Q),fontsize=15)
          ax1.legend(frameon=False,fontsize=12)
          
          ax2.plot(np.arange(0,N), y-pred, 'r',linewidth=2) 
          ax2.grid(True)
          ax2.set_ylabel('errors',fontsize=15) 
          ax2.set_xlabel('NMSE = ' + str(NMSE),fontsize=15)

    return Cest, Kest, pred, NMSE

class ModularVolterra():
  def __init__(self, isLET=True, isPlot=False):
    self.isLET = isLET
    self.isPlot = isPlot

  def function_LET1Q2_kernelestimate(self, Cest, Laguerre, L, Q): 

    M = max(np.shape(Laguerre))

    Kest = {}
    Kest["k0"] = Cest["c0"] 
    k1 = Laguerre @ Cest["c1"]
    Kest["k1"] = k1

    r = len(Laguerre[:,0])

    if Q==2: 
        c2 = Cest['c2']
        temp = np.zeros((M, M)) 
        for k1 in range(L):
            for k2 in range(L):
                temp = temp + c2[k1,k2] * Laguerre[:,k1:k1+1] @ Laguerre[:,k2:k2+1].T;   
        Kest['k2'] = temp
        del temp

    return Kest

  def function_LET1Q2_coeffestimate(self, y, VV, L, Q, Mdiscard):

    y = y.flatten()
    N = len(y) 
    y = y[Mdiscard:N] 

    VVorg = VV
    VV = VV[Mdiscard:N,:]

    XtX = VV.T @ VV
    EV, ED = np.linalg.eig(XtX)

    TH_sv = 1e-6
    if min(abs(np.diag(ED)))>TH_sv:
        ALL_C = np.linalg.inv(XtX) @ VV.T @ y
    else:
        [Usvd, Ssvd, Vsvd] = np.linalg.svd(VV)
        Vsvd = Vsvd.T
        SV = abs(np.diag(Ssvd))
        Idx_sv = int(np.nonzero(SV>=TH_sv)[-1][-1] + 1) # check
        Pseudo_invVV = Vsvd[:,0:Idx_sv] @ np.diag(1/SV[0:Idx_sv]) @ Usvd[:,0:Idx_sv].T
        ALL_C = Pseudo_invVV @ y

    yorg_predict = VVorg @ ALL_C
    y_predict = VV @ ALL_C
    nmse = np.mean((y - y_predict)**2) / np.mean(y**2)

    Cest = {}
    tempC = ALL_C
    Cest["c0"] = tempC[0]
    tempC = tempC[1:]

    c1 = tempC[0:L]
    tempC = tempC[L:]
    Cest["c1"] = c1

    if Q==2:
          c2 = np.zeros((L,L))
          mmm = 0
          for kk in range(L):
              c2[kk,kk:L] = tempC[mmm:mmm+L-kk].T
              mmm = mmm + L - kk # + 1 
          c2 = (c2 + c2.T) / 2
          Cest['c2'] = c2

    return Cest, yorg_predict, nmse

  def function_generate_laguerre(self, alpha, L, M):
      
      mm = np.arange(0,M-1)
      beta = 1 - alpha
      rootAlpha = np.sqrt(alpha)
      L_buf = np.sqrt((alpha**mm) * beta)

      mmm = np.zeros((M,1))
      Laguerre = np.zeros((len(L_buf),L))
      Laguerre[:,0] = L_buf

      for n1 in range(0,L-1):
          for n2 in mm.tolist():
              mmm[n2,0] = L_buf[n2]
              if n2==0:
                L_buf[n2] = rootAlpha * L_buf[n2]
              else:
                L_buf[n2] = rootAlpha * (L_buf[n2-1] + mmm[n2,0]) - mmm[n2-1,0]
          Laguerre[:,n1+1] = L_buf

      return Laguerre

  def function_Q2self_each_input(self, V):

      N = np.shape(V)[0]
      L = np.shape(V)[1] 

      Nxx = int(L * (L + 1 )/ 2)
      Vxx = np.zeros((N, Nxx))

      cnt = -1
      for k1 in range(L):
          for k2 in range(k1,L):
              cnt = cnt + 1 
              Vxx[:,cnt] = V[:,k1] * V[:,k2]

      return Vxx

  def function_Q1_each_input(self, x, Laguerre):

      N = len(x)
      L = np.shape(Laguerre)[1] 

      Vx = np.zeros((N, L))
      for k in range(L):
          v_column = np.convolve(Laguerre[:,k], x.flatten())   
          Vx[:,k] = v_column[0:N]

      return Vx

  def function_LET1Q2_LFparameters(self, x, y, LETparameters):

    alpha = LETparameters["alpha"]
    L = LETparameters["L"]
    M = LETparameters["M"]
    Q = LETparameters["Q"]
    Mdiscard = LETparameters["Mdiscard"] 

    x = x.flatten()
    y = y.flatten()

    Laguerre = self.function_generate_laguerre(alpha, L, M)

    V0 = np.ones((len(x), 1))
    V1 = self.function_Q1_each_input(x, Laguerre)
    V2 = self.function_Q2self_each_input(V1)

    VV_est = np.concatenate((np.concatenate((V0,V1),axis=1),V2),axis=1)

    Cest, y_predict, nmse = self.function_LET1Q2_coeffestimate(y, VV_est, L, Q, Mdiscard)
    
    Kest = self.function_LET1Q2_kernelestimate(Cest, Laguerre, L, Q)

    return nmse, Cest, Kest, y_predict

  def function_LET1Q2(self, x, y, alpha, L, M, Nfig):

    x = x.flatten()
    y = y.flatten()
    N = len(x)

    Mdiscard = M
    Q = 2

    LETparameters = {}
    LETparameters["alpha"] = alpha 
    LETparameters["Q"] = Q
    LETparameters["L"] = L
    LETparameters["M"] = M
    LETparameters["Mdiscard"] = Mdiscard

    NMSE, Cest, Kest, prediction_LET1Q2 = self.function_LET1Q2_LFparameters(x, y, LETparameters)

    if self.isPlot:          
                
      fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,6))
      ax1.plot(np.arange(0,M-1), Kest["k1"], linewidth = 2) 
      ax1.set_title('LET1 analysis: alpha = ' + str(alpha) + ', L = ' + str(L) + ' and Q = 2') 
      ax1.grid(True)
      ax1.set_ylabel('1$^{st}$-order kernel')
      xx, yy = np.meshgrid(np.arange(0,M-1), np.arange(0,M-1))
      ax2 = plt.axes(projection='3d')
      ax2.contour3D(xx, yy, np.reshape(np.meshgrid(np.array(Kest['k2'])),(M-1,M-1)), 200)
      ax2.set_zlabel('2$^{nd}$-order kernel')

      fig, ax1 = plt.subplots(1,1,figsize=(6,6))
      ax1.plot(np.arange(0,N), y, 'b', linewidth=2, label = 'true output')
      ax1.plot(np.arange(0,N), prediction_LET1Q2, 'k--', linewidth=2, label = 'estimated output')  
      ax1.set_title('LET1 analysis: alpha = ' + str(alpha) + ', L = ' + str(L) + ' and Q = 2')
      ax1.set_xlabel('NMSE = ' + str(NMSE)) 

    return Kest

  def function_LET1Q2_PDMs(self, x, Kest, L, Nfig, inputID, Npdms_input=3):

    rms = {}
    rms["x1"] = np.mean((x-np.mean(x))**2) 
    Mtx_kernels = np.concatenate((np.expand_dims(Kest["k1"], axis=1), rms["x1"]*Kest["k2"]), axis=1)
    Uk, Sk, Vk = np.linalg.svd(Mtx_kernels)
    Vk = Vk.T
    for kk in range(Uk.shape[1]):
        temp = Uk[:,kk]
        if sum(temp)<0:
          Uk[:,kk] = -Uk[:,kk]
          
    Sk = np.diag(Sk)

    if self.isPlot:

      fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
      ax1.plot(Sk)
      ax1.grid(True)
      if inputID is None: 
          ax1.set_title('singular values (S_k) of the kernel matrix') 
      else:
          ax1.set_title('singular values (S_k) of the kernel matrix ' + str(inputID)) 
      ax1.set_ylabel('S_k')
            
    MM = len(Sk)
    sum_of_Sk = np.mean(list(np.cumsum(Sk,axis=0))/sum(Sk),axis=1) # np.cumsum(Sk)/sum(Sk)

    if self.isPlot:

      ax2.plot(np.arange(1,MM+1), sum_of_Sk, 'b', linewidth = 3)
      ax2.plot([1, MM], [0.9, 0.9], 'r--', linewidth = 3)
      ax2.grid(True) 
      ax2.set_ylabel('cumulative S_k')

    Npdms = Npdms_input
    if Npdms is None:
        Npdms = 1

    if Npdms>L:
        Npdms = L 

    PDMs = Uk[:,0:Npdms]

    if self.isPlot:

      fig, ax = plt.subplots(Npdms,1,figsize=(5,5)) 
      for kk in range(Npdms):
          ax[kk].plot(np.arange(0,len(Kest["k1"])), PDMs[:,kk], linewidth = 2)
          ax[kk].grid(True)
          if kk==1:
            ax[kk].set_title('PDMs ' + str(inputID)) 
            ax[kk].set_ylabel('PDM #' + str(kk))

    return PDMs, Npdms

  def function_LET1PDMoutputs(self, x, PDMs):
                                                          
    x = x.flatten()
    N = len(x) 

    Npdms = PDMs.shape[1]                                                     
    uu = np.zeros((N, Npdms))
    for kk in range(Npdms):
        temp = np.convolve(x, PDMs[:,kk])
        uu[:,kk] = temp[0:N]

    return uu

  def function_estimate_3rd_order_ANFs(self, x, y, PDMs):

    x = x.flatten()
    y = y.flatten()
    N = len(y)
    Mdiscard = PDMs.shape[0] 

    uu = self.function_LET1PDMoutputs(x, PDMs)
        
    ANFs_order = 3 
    Npdms = PDMs.shape[1]

    ANFs_coeff = np.zeros((ANFs_order * Npdms, 1)) 
    Nunknowns1 = ANFs_coeff.shape[0]

    Mtx = np.zeros((N, Nunknowns1)) 
    cnt = -1
    for mmm in range(Npdms):
        for kk in range(ANFs_order):
            cnt = cnt + 1
            Mtx[:,cnt] = uu[:,mmm]**kk

    Mtx = np.concatenate((np.ones((N, 1)), Mtx), axis = 1)

    Mtx_org = Mtx
    Mtx = Mtx[Mdiscard:N,:]
    y = y[Mdiscard:N]

    EV, ED = np.linalg.eig(Mtx.T @ Mtx)
    min(abs(np.diag(ED)));
    TH_sv = 1e-2 
    if min(abs(np.diag(ED)))>TH_sv:
        ANFs_coeff = np.linalg.inv(Mtx.T @ Mtx) @ Mtx.T @ y
    else:
        [Usvd, Ssvd, Vsvd] = np.linalg.svd(Mtx)
        Vsvd = Vsvd.T
        SV = np.diag(Ssvd)
        Idx_sv = int(np.nonzero(SV>=TH_sv)[-1][-1] + 1)
        Pseudo_invMTX = Vsvd[:,0:Idx_sv] @ np.diag(np.diag(1/SV[0:Idx_sv])) @ Usvd[:,0:Idx_sv].T 
        ANFs_coeff = Pseudo_invMTX @ y

    yest = {}
    yest["all"] = Mtx_org @ ANFs_coeff
    yest["no_transition"] = Mtx @ ANFs_coeff
    nmse = np.mean((y - yest["no_transition"])**2) / np.mean(y**2)

    ANFs = {}
    ANFs["const"] = ANFs_coeff[0]
    ANFs_coeff = ANFs_coeff[1:]
    for mmm in range(Npdms):
        namestr = "pdm" + str(mmm)
        ANFs[namestr] = ANFs_coeff[0:ANFs_order]
        ANFs_coeff = ANFs_coeff[ANFs_order:]

    return yest, nmse, ANFs, uu

  def function_plot_nlANFs(self, uu, PDMs, ANFs, Nfig):

    ANFs_order = 3 
    Npdms = PDMs.shape[1]
    M = PDMs.shape[0]
    wgt_domain = 1
    domain = {}
    ranges = {}

    for mmm in range(Npdms):
        NLcoeff = ANFs["pdm" + str(mmm)]
        uu_bound = int(np.ceil(wgt_domain*np.std(uu[:,mmm])))
        min_domain = -uu_bound
        max_domain = uu_bound
        in_domain = np.arange(min_domain, max_domain, 0.01)+np.mean(uu[:,mmm])
        out_range = np.zeros(in_domain.shape)
        for kk in range(ANFs_order):
            out_range = out_range + NLcoeff[kk] * in_domain**kk
        domain["pdm" + str(mmm)] = in_domain
        ranges["pdm" + str(mmm)] = out_range    

    xmin = np.zeros((Npdms, 1))
    xmax = np.zeros((Npdms, 1))

    if self.isPlot:

      fig, ax = plt.subplots(1,1)
      for mmm in range(Npdms):

          x_domain = domain["pdm" + str(mmm)]
          xmin[mmm, 0] = min(x_domain)
          xmax[mmm, 0] = max(x_domain)
          y_range = ranges["pdm" + str(mmm)]

          montages = ['b', 'g', 'r', 'k', 'm', 'c', 'y', 'b--', 'g--']

          ax.plot(x_domain, y_range, montages[mmm], linewidth=3, label = 'PDM #'+str(mmm+1))
      
      ax.set_title('ANFs')
      ax.grid(True) 
      ax.set_xlabel('u')
      ax.set_ylabel('z')

    return True

  def PDM_1(self, x, y, alpha, L, Nfig, Npdms_input=3):

    x = x.flatten()
    y = y.flatten()
    if len(x) != len(y): 
        raise NameError('lengths of input and output are different') 

    N = len(x)

    if alpha<=0 or alpha>=1:
        print('alpha should be between 1 and 9, default vlaue alpha = 0.5') 
        alpha = 0.5

    M = (-30 - np.log(1-alpha)) / np.log(alpha)
    M = int(np.ceil(M))

    if L<=0 or L>9: 
        print('L should be between 1 and 9, default vlaue L = 5')
        L = 5

    if Nfig is None: 
        Nfig = 1
          
    Kest = self.function_LET1Q2(x, y, alpha, L, M, Nfig) 
  
    PDMs, Npdms = self.function_LET1Q2_PDMs(x, Kest, L, Nfig+1, None, Npdms_input=Npdms_input)    

    pred, NMSE, ANFs, uu = self.function_estimate_3rd_order_ANFs(x, y, PDMs)
    self.function_plot_nlANFs(uu, PDMs, ANFs, Nfig+3)

    if self.isPlot:

      fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
      ax1.plot(np.arange(0,N), y, 'b', linewidth=2, label = 'output signal')
      ax1.plot(np.arange(0,N), pred['all'], 'k--', linewidth=2, label = 'model prediction')
      ax1.grid(True)
      ax1.legend() 
      ax1.set_title('PDM and ANF analysis: ' + str(Npdms) + ' PDMs') 
      ax2.plot(np.arange(0,N), y-pred['all'], 'r', 'linewidth', 2)
      ax2.grid(True)
      ax2.set_ylabel('residual')
      ax2.set_xlabel('nmse = ' + str(NMSE)) 

    pred = pred['all'] 

    return Npdms, PDMs, ANFs, pred, NMSE