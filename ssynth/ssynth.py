#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:57:23 2021

@author: cristian
"""


import numpy as np
import matplotlib.pylab as plt
from astropy import constants as C
import math
from astropy.io import fits
from astropy.io import fits
from astropy import units as u
from astropy.io.fits.verify import VerifyWarning
from PyAstronomy import pyasl
from specutils import Spectrum1D
from specutils.manipulation import SplineInterpolatedResampler
from scipy import signal
from astropy.constants import c
from scipy.optimize import curve_fit
from scipy.fft import fft
from specutils.fitting import fit_continuum
from matplotlib import gridspec
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Chebyshev1D
from numba import jit
from os import scandir, getcwd
from os.path import abspath
from progress.bar import ChargingBar
spline3 = SplineInterpolatedResampler()
plt.ion()




def help():
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n')
    print('                  CREATE SPECTRA SYNTHETIC DATASET')
    print('                      Mar 2024 - v1.01 \n\n\n')   


def create(sa, sb, ma, mb, la, lb, n, p, yran, i=90, err=0., snr=10, vgamma=0, T0=2455000,obj='NoObject'):
    VerifyWarning('ignore')
    spline3 = SplineInterpolatedResampler()
    nstr=int(np.log10(n))
    sa=sa.replace('.fits','')
    sb=sb.replace('.fits','')
    #ha = fits.open(sa+'.fits', 'update')
    #hb = fits.open(sb+'.fits', 'update')
    wa,fa = pyasl.read1dFitsSpec(sa+'.fits')
    wb,fb = pyasl.read1dFitsSpec(sb+'.fits')
    #delta = min(ha[0].header['CDELT1'],hb[0].header['CDELT1'])
    delta=0.01
    winf=max(wa[0],wb[0])+50
    wsup=min(wa[-1],wb[-1])-50
    new_disp_grid = np.arange(winf,wsup,delta)
    KA=(2*math.pi*C.G.value*C.M_sun.value/(86400*p))**(1/3)*mb*np.sin(math.radians(i))/(ma+mb)**(2/3)/1000
    KB=(2*math.pi*C.G.value*C.M_sun.value/(86400*p))**(1/3)*ma*np.sin(math.radians(i))/(ma+mb)**(2/3)/1000
    for j in range(n):
        out='obs'+str(j+1).zfill(nstr+2)
        print('Writing spectrum',out+'.fits')
        xjd=T0+np.random.normal()*yran*365.25
        vra=KA*np.sin(2*math.pi/p*(xjd-T0))+vgamma
        wlprime_A = wa * np.sqrt((1.+vra/299792.458)/(1.-vra/299792.458))
        aux_sa = Spectrum1D(flux=fa*u.Jy, spectral_axis=wlprime_A *0.1*u.nm)
        aux2_sa = spline3(aux_sa, new_disp_grid*0.1*u.nm)
        fa_dop = aux2_sa.flux.value
        fa_dop =  splineclean(fa_dop)
        vrb=-KB*np.sin(2*math.pi/p*(xjd-T0))+vgamma
        wlprime_B = wb * np.sqrt((1.+vrb/299792.458)/(1.-vrb/299792.458))
        aux_sb = Spectrum1D(flux=fb*u.Jy, spectral_axis=wlprime_B *0.1*u.nm)
        aux2_sb = spline3(aux_sb, new_disp_grid*0.1*u.nm)
        fb_dop = aux2_sb.flux.value
        fb_dop =  splineclean(fb_dop)
        fout=la*fa_dop+lb*fb_dop
        fout[fout<0]=0
        fout2=np.random.normal(fout,np.sqrt(fout)*np.sqrt(np.mean(fout))/snr,len(fout))
        pyasl.write1dFitsSpec(out+'.fits', fout2, wvl=new_disp_grid, clobber=True)
        hdul = fits.open(out+'.fits', 'update')
        hdul[0].header['HJD'] = round(xjd,6)
        hdul[0].header['VRA'] = round(np.random.normal(vra,err),2)
        hdul[0].header['VRB'] = round(np.random.normal(vrb,err),2)
        hdul[0].header['OBJECT'] = obj
        hdul.flush(output_verify='ignore')
        hdul.close(output_verify='ignore')

def amp(ma,mb,p):
    K=(2*math.pi*C.G.value*C.M_sun.value/(86400*p))**(1/3)*mb*np.sin(math.radians(90))/(ma+mb)**(2/3)/1000
    print('K for primary: ',round(K,2),' km/s')
    
    
def fxcor(img, tmp, rvrange=2000,order=10,wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900'):
    #leer espectros
    wimg,fimg = pyasl.read1dFitsSpec(img)
    wta,fta = pyasl.read1dFitsSpec(tmp)
    #ajustar continuo
    fci=continuum(w=wimg, f=fimg, type='diff', graph=False)
    fct=continuum(w=wta, f=fta, type='diff', graph=False)
    #leer header
    hdul = fits.open(tmp, 'update')
    delta = hdul[0].header['CDELT1']
    nombre = img.replace('.fits','')
    winf=max(wimg[0],wta[0])
    wsup=min(wimg[-1],wta[-1])
    if wreg==None:
        wreg=str(int(max(wimg[0],wta[0])))+'-'+str(int(min(wimg[-1],wta[-1])))
    #crear grilla de w optima
    new_disp_grid,fmask = setregion(wreg,delta,winf,wsup)
    #convertir a escala log w
    aux_grid=np.log(new_disp_grid)
    #delta ln w es tomado al extremo de rojo
    dlog = aux_grid[-1] - aux_grid[-2]
    #nueva grilla en escala logartimica y delta discreto
    new_log_grid=np.arange(aux_grid[0],aux_grid[-1],dlog)
    #reescalar mascara a nueva grilla log
    aux_fmask = Spectrum1D(flux=fmask*u.Jy, spectral_axis=np.log(new_disp_grid)*0.1*u.nm)
    aux2_fmask= spline3(aux_fmask, new_log_grid*0.1*u.nm)
    log_fmask = splineclean(aux2_fmask.flux.value)
    #reescalar espectro a nueva grilla log
    aux_img = Spectrum1D(flux=fci*u.Jy, spectral_axis=np.log(wimg)*0.1*u.nm)
    aux2_img = spline3(aux_img, new_log_grid*0.1*u.nm)
    log_img=splineclean(aux2_img.flux.value)
    fi2=log_img*log_fmask
    #reescalar template a nueva grilla log
    aux_sa = Spectrum1D(flux=fct*u.Jy, spectral_axis=np.log(wta)*0.1*u.nm)
    aux2_sa = spline3(aux_sa, new_log_grid*0.1*u.nm)
    log_tmp = splineclean(aux2_sa.flux.value)
    ft2=log_tmp*log_fmask
    #calcular correlaciones
    cc1=signal.correlate(fi2,ft2,method='fft')
    #normalizar correlaciones
    cc1=cc1/(np.sqrt(np.sum(np.power(fi2,2)))*(np.sqrt(np.sum(np.power(ft2,2)))))
    #encontrar valor de pico cc1
    i1=int(np.where(cc1==max(cc1))[0])
    #armar eje X: lags (delta ln w)
    lamlog1=new_log_grid-new_log_grid[0]
    lamlog2=-new_log_grid+new_log_grid[0]
    lamlog2.sort()
    llog=np.concatenate((lamlog2[0:-1],lamlog1),axis=0)
    axisrv=llog*c.value/1000
    #ajustar funcion
    ibmax=np.argmax(cc1)
    sfit=False
    try:
        xb=axisrv[ibmax-10:ibmax+10]
        yb=cc1[ibmax-10:ibmax+10]
        ygb1=np.mean(cc1)
        ygb2=ygb1
        mb = np.sum(xb * (yb-(ygb1+ygb2)/2)) / np.sum(yb-(ygb1+ygb2)/2)
        sigb = np.sqrt(np.abs(np.sum((yb-(ygb1+ygb2)/2) * (xb - mb)**2) / np.sum(yb-(ygb1+ygb2)/2)))
        pb1,pb2 = curve_fit(Gauss, xb, yb-(ygb1+ygb2)/2, p0=[np.max((yb-(ygb1+ygb2)/2)), mb, sigb])
        ybgauss=Gauss(xb, *pb1)+(ygb1+ygb2)/2
        xrv=pb1[1]
        rverr = np.sqrt(np.diag(pb2))[1]
        sfit=True
    except Exception:
        xrv=axisrv[ibmax]
        sfit=False
        err_g1=str(np.nan)
        err_g2=str(np.nan)
    #graficar
    f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 5]})
    plt.setp(a0.get_yticklabels(), visible=False)
    plt.setp(a0.get_xticklabels(), visible=False)
    a0.yaxis.offsetText.set_visible(False)
    a0.plot(axisrv,cc1,color='blue')
    a0.axvline(axisrv[i1], color='red', linestyle='--',linewidth=1)
    a1.set_xlabel("Radial Velocity [km/s]", fontsize=10)
    a1.set_ylabel("Correlation", fontsize=10)
    a1.plot(axisrv,cc1,color='blue')
    a1.set(xlim=(axisrv[i1-15], axisrv[i1+15]))
    ymin2=min(cc1[i1-15:i1+15])
    ymax2=max(cc1[i1-15:i1+15])
    ex2=(ymax2-ymin2)*0.1
    a1.set(ylim=(ymin2-ex2, ymax2+ex2))
    if sfit:
        plt.plot(xb, yb, color='black',marker='.',linestyle='')
        plt.plot(xb, ybgauss, color='green', label='fit',linestyle='--')
        iran=int(rvrange*1000/(c.value*dlog))
        #cropcc regions with rvrange parameter
        reg2=axisrv[i1-iran:i1+iran]
        fcc2=cc1[i1-iran:i1+iran]
        #filter low frequencies in cc function
        ccnew=continuum(w=reg2, f=fcc2, order=order, type='diff', graph=False)
        # estimate sigma antisymmetric noise
        cc1_neg=ccnew[0:iran]
        cc1_pos=ccnew[iran:iran*2]
        dif_neg = cc1_neg - cc1_pos[::-1]
        dif_pos = cc1_pos - cc1_neg[::-1]
        cc_anti = np.concatenate((dif_neg,dif_pos))
        sig_anti = np.std(cc_anti)
        r = np.max(ccnew)/(np.sqrt(2*sig_anti))
        plt.tight_layout()
        plt.savefig(nombre+'_CC.jpg')
        #graphicate cc function
        fig1, (ax1,ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[5, 1]})
        ax1.plot(reg2,ccnew,color='blue')
        ax2.plot(reg2,cc_anti,color='red')
        plt.tight_layout()
        plt.savefig(nombre+'_err.jpg')
        #calculate FFT
        wfreq1=fft(ccnew)
        nmed=int(len(wfreq1.imag)/4)
        ys2=np.log(np.sqrt(wfreq1.real[1:nmed]**2+wfreq1.imag[1:nmed]**2))
        ys2=ys2-ys2[-1]
        xs2=np.arange(1,nmed,1)
        # fit gaussian function without wieghts
        aux_x=np.concatenate((-1*xs2[::-1],xs2))
        aux_y=np.concatenate((ys2[::-1],ys2))
        az=np.where(aux_y==0)
        aux_y[az]=1e-10
        p1, pc1 = curve_fit(dobuleG, aux_x, aux_y)
        y_gau1 = dobuleG(xs2, *p1)
        # fit weighted gaussian function with relative sigma
        noise_sigma=1/aux_y/np.max(aux_y)
        p2, pc2 = curve_fit(dobuleG, aux_x, aux_y,sigma=noise_sigma,absolute_sigma=False)
        y_gau2 = dobuleG(xs2, *p2)
        #graphicate fft function
        plt.figure()
        plt.plot(xs2,ys2,color='gray',marker='.',ls='')
        plt.plot(xs2,y_gau1,ls='-',marker='',color='red')
        plt.plot(xs2,y_gau2,ls='-',marker='',color='blue')
        #error for gaussian at half maximum
        B_exp1 = p1[1]*2.35482/2
        plt.axvline(B_exp1,color='red',ls='--')
        plt.xlim((0,B_exp1*3))
        B_exp2 = p2[1]*2.35482/2
        plt.axvline(B_exp2,color='blue',ls='--')
        plt.legend(('FFT orders','Not weighted','Weighted','Freq limit 1','Freq limit 2'))
        err_g1 = len(wfreq1)/(16*B_exp1*(1+r))+rverr
        err_g2 = len(wfreq1)/(16*B_exp2*(1+r))+rverr
    else:
        plt.legend(('No fit'))
    plt.tight_layout()
    plt.savefig(nombre+'_FFT.jpg')
    print('RV (not weighted): '+str(round(xrv,3))+' +/- '+str(round(err_g1,3))+' km/s')
    print('RV (weighted): '+str(round(xrv,3))+' +/- '+str(round(err_g2,3))+' km/s')

def fxcompare(img1, img2, tmp, rvrange=2000, order=12, ystep=0, wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900'):
    #leer espectros
    wimg1,fimg1 = pyasl.read1dFitsSpec(img1)
    wimg2,fimg2 = pyasl.read1dFitsSpec(img2)    
    wta,fta = pyasl.read1dFitsSpec(tmp)
    #ajustar continuo
    fci1=continuum(w=wimg1,f=fimg1,  order=order, type='diff', graph=False)
    fci2=continuum(w=wimg2, f=fimg2,  order=order, type='diff', graph=False)    
    fct=continuum(w=wta, f=fta,  order=order, type='diff', graph=False)
    #leer header
    hdul = fits.open(tmp, 'update')
    delta = hdul[0].header['CDELT1']
    winf=max(wimg1[0],wta[0])
    wsup=min(wimg1[-1],wta[-1])
    #crear grilla de w optima
    new_disp_grid,fmask = setregion(wreg,delta,winf,wsup)
    #convertir a escala log w
    aux_grid=np.log(new_disp_grid)
    #delta ln w es tomado al extremo de rojo
    dlog = aux_grid[-1] - aux_grid[-2]
    #nueva grilla en escala logartimica y delta discreto
    new_log_grid=np.arange(aux_grid[0],aux_grid[-1],dlog)
    #reescalar mascara a nueva grilla log
    aux_fmask = Spectrum1D(flux=fmask*u.Jy, spectral_axis=np.log(new_disp_grid)*0.1*u.nm)
    aux2_fmask= spline3(aux_fmask, new_log_grid*0.1*u.nm)
    log_fmask = splineclean(aux2_fmask.flux.value)
    #reescalar espectro a nueva grilla log
    aux_img1 = Spectrum1D(flux=fci1*u.Jy, spectral_axis=np.log(wimg1)*0.1*u.nm)
    spl_img1 = spline3(aux_img1, new_log_grid*0.1*u.nm)
    log_img1=splineclean(spl_img1.flux.value)
    aux_img2 = Spectrum1D(flux=fci2*u.Jy, spectral_axis=np.log(wimg2)*0.1*u.nm)
    spl_img2 = spline3(aux_img2, new_log_grid*0.1*u.nm)
    log_img2=splineclean(spl_img2.flux.value)
    fi1=log_img1*log_fmask
    fi2=log_img2*log_fmask    
    #reescalar template a nueva grilla log
    aux_sa = Spectrum1D(flux=fct*u.Jy, spectral_axis=np.log(wta)*0.1*u.nm)
    aux2_sa = spline3(aux_sa, new_log_grid*0.1*u.nm)
    log_tmp = splineclean(aux2_sa.flux.value)
    ft=log_tmp*log_fmask
    #graficar espectros normalizados
    plt.figure()
    plt.plot(np.e**new_log_grid,fi1/(np.max(fi1)-np.min(fi1))+1,color='red',ls='--',  linewidth=1, marker='',label=img1)
    plt.plot(np.e**new_log_grid,fi2/(np.max(fi2)-np.min(fi2))+1+ystep,color='blue',ls='--', linewidth=1, marker='',label=img2)
    plt.plot(np.e**new_log_grid,ft/(np.max(ft)-np.min(ft))+1,color='black',ls='-', linewidth=1, marker='',label=tmp)    
    plt.legend()
    plt.plot(np.e**new_log_grid,ft/(np.max(ft)-np.min(ft))+1+ystep,color='black',ls='-', linewidth=1, marker='',label=tmp)   
    plt.tight_layout()
    #calcular correlaciones
    cc1=signal.correlate(fi1,ft,method='fft')
    cc2=signal.correlate(fi2,ft,method='fft')    
    #normalizar correlaciones
    cc1=cc1/(np.sqrt(np.sum(np.power(fi1,2)))*(np.sqrt(np.sum(np.power(ft,2)))))
    cc2=cc2/(np.sqrt(np.sum(np.power(fi2,2)))*(np.sqrt(np.sum(np.power(ft,2)))))    
    #encontrar valor de pico cc1
    i1=int(np.where(cc1==max(cc1))[0])
    i2=int(np.where(cc2==max(cc2))[0])    
    #armar eje X: lags (delta ln w)
    lamlog1=new_log_grid-new_log_grid[0]
    lamlog2=-new_log_grid+new_log_grid[0]
    lamlog2.sort()
    llog=np.concatenate((lamlog2[0:-1],lamlog1),axis=0)
    axisrv=llog*c.value/1000
    #ajustar funcion
    i1max=np.argmax(cc1)
    i2max=np.argmax(cc2)   
    iran=int(rvrange*1000/(c.value*dlog))
    #cropcc regions with rvrange parameter
    regx=axisrv[i1-iran:i1+iran]
    fcc1=cc1[i1-iran:i1+iran]
    fcc2=cc2[i1-iran:i1+iran]    
    ccnew1=continuum(w=regx, f=fcc1, order=4, type='diff', graph=False)
    ccnew2=continuum(w=regx, f=fcc2, order=4, type='diff', graph=False)    
    #graficar correlaciones
    plt.figure()
    plt.plot(regx,ccnew1,color='red',label=img1)
    plt.plot(regx,ccnew2,color='blue',label=img2)    
    plt.xlabel("Radial Velocity [km/s]", fontsize=10)
    plt.ylabel("Correlation", fontsize=10)
    plt.legend()



def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def dobuleG(x, a, sigma):
    return a * np.exp(-(x)**2 / (2 * sigma**2))

def continuum(w,f, order=12, type='fit', lo=2, hi=3, nit=10, graph=True):
    w_cont=w.copy()
    f_cont=f.copy()
    sigma0=np.std(f_cont)
    wei=~np.isnan(f_cont)*1
    i=1
    nrej1=0
    while i < nit:
        c0=np.polynomial.chebyshev.Chebyshev.fit(w_cont,f_cont,order,w=wei)(w_cont)
        resid=f_cont-c0
        sigma0=np.sqrt(np.average((resid)**2, weights=wei))
        wei = 1*np.logical_and(resid>-lo*sigma0,resid<sigma0*hi)
        nrej=len(wei)-np.sum(wei)
        if nrej==nrej1:
            break
        nrej1=nrej
        i=i+1
    s1=Spectrum1D(flux=c0*u.Jy, spectral_axis=w_cont*0.1*u.nm) 
    c1= fit_continuum(s1, model=Chebyshev1D(order),fitter=LinearLSQFitter())
    if type=='fit':
        fout=c1(w*0.1*u.nm).value
    elif type=='ratio':
        fout=f_cont/c1(w*0.1*u.nm).value
    elif type=='diff':
        fout=f_cont-c1(w*0.1*u.nm).value
    if graph:
        fig = plt.figure(figsize=[20,10])
        ngrid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[5, 1])
        ax1 = fig.add_subplot(ngrid[0])
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel('Flux')
        ax1.plot(w,f,color='gray')
        ax1.plot(w_cont,f_cont,color='blue',linestyle='',marker='.',markersize=2)
        ax1.plot(w_cont,c1(w_cont*0.1*u.nm).value,c='r',linestyle='--')
        ax2 = fig.add_subplot(ngrid[1], sharex=ax1,sharey=ax1)
        plt.xlabel('Wavelength [nm]')
        ax2.plot(w,(f-c1(w*0.1*u.nm).value),color='gray',linestyle='',marker='.',markersize=1)
        ax2.plot(w_cont,(f_cont-c1(w_cont*0.1*u.nm).value),color='blue',linestyle='',marker='.',markersize=1)
        ax2.axhline(y=0, color='red', linestyle='--',linewidth=1)
        plt.tight_layout()
        plt.show()
    return(fout)
    

def setregion(wreg,delta,winf,wsup,amort=0.1):
    reg1=wreg.split(',')
    reg2=[]
    for i,str1 in enumerate(reg1):
        reg2.append([int(str1.split('-')[0]),int(str1.split('-')[1])])
    reg3=[]
    stat1=True
    for j,wvx in enumerate(reg2):
        x1=wvx[0]
        x2=wvx[1]
        if stat1:
            if x1>=winf:
                reg3.append(wvx)
                stat1=False
            elif x1<winf and x2>winf:
                wvx[0]=winf
                reg3.append(wvx)
                stat1=False
            elif x1<winf and x2<=winf:
                stat1=True
        else:
            if x1>wsup and x2>wsup:
                break
            elif x1<wsup and x2>=wsup:
                wvx[1]=wsup
                reg3.append(wvx)
            elif x1<wsup and x2<wsup:
                reg3.append(wvx)
    wvl=np.arange(reg3[0][0],reg3[-1][1]+delta,delta)
    f=np.zeros(len(wvl))
    for k,interv in enumerate(reg3):
        x1=interv[0]
        x2=interv[1]
        i1 = np.abs(wvl - x1).argmin(0)
        i2 = np.abs(wvl - x2).argmin(0)
        am2=amort*(x2-x1)
        xarr=wvl[i1:i2]
        mask=np.zeros(len(xarr))
        for k,w in enumerate(xarr):
            if w<=(x1+am2):
                mask[k]=np.sin(np.pi*(w-x1)/(2*am2))
            elif w>(x1+am2) and w<(x2-am2):
                mask[k]=1
            else:
                mask[k]=np.cos(np.pi*(w-x2+am2)/(2*am2))
        f[i1:i2]=mask
    return(wvl,f)
    
    
@jit(nopython=True)
def splineclean(fspl):
    if np.isnan(fspl[0]):
        cinit=1
        while np.isnan(fspl[cinit]):
            cinit+=1
        finit=fspl[cinit]
        fspl[0:cinit+1]=finit
    if np.isnan(fspl[-1]):
        cend=-2
        while np.isnan(fspl[cend]):
            cend-=1
        fend=fspl[cend]
        fspl[cend:len(fspl)]=fend
    return(fspl)
    
    
def specfit(img, lit, order=50, lo=2, hi=4, snr=None, wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900'):
    plt.ion()
    print('\n\tRunning SPECFIT\n')
    VerifyWarning('ignore')
    #leer espectro
    wimg,fimg = pyasl.read1dFitsSpec(img)
    hdul = fits.open(img, 'update')
    d1 = hdul[0].header['CDELT1']
    #leer lista de templates
    if lit[len(lit)-1] == '/':
        lit=lit[0:len(lit)-1]
    ltemp=sorted(listmp(lit))
    # leer primero template 
    waux2,faux2 = pyasl.read1dFitsSpec(ltemp[0])
    htmp = fits.open(ltemp[0], 'update')
    d2 = htmp[0].header['CDELT1']
    #generar nueva grilla lineal
    gap=50
    winf=max(wimg[0],waux2[0])+gap
    wsup=min(wimg[-1],waux2[-1])-gap
    new_disp_grid,fmask = setregion(wreg,max(d1,d2),winf,wsup)
    # resampleo del espectro
    cont1=continuum(wimg, fimg,  order=order, nit=10, type='fit', lo=lo,hi=hi, graph=False)
    spec_cont=fimg-cont1
    aux_img = Spectrum1D(flux=spec_cont*u.Jy, spectral_axis=wimg*0.1*u.nm)
    aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
    new_flux = splineclean(aux2_img.flux.value)*fmask
    norma1=np.mean(new_flux**2)
    # leer lista de templates (filtering and apodizing)
    matrix_tmp=np.zeros(shape=(len(ltemp),len(new_disp_grid)))
    vector_t=np.zeros(len(ltemp))
    sigma=np.zeros(len(ltemp))
    bar1 = ChargingBar('Processing templates:', max=len(ltemp))
    for j,tmp in enumerate(ltemp):
        htmp = fits.open(tmp, 'update')
        teff= htmp[0].header['TEFF']
        vector_t[j]=teff
        htmp.close(output_verify='ignore')
        wt1,ft1 = pyasl.read1dFitsSpec(tmp)
        temp_cont = continuum(wt1, ft1, order=order, nit=10, type='diff', lo=lo,hi=hi, graph=False)
        aux_tmp1 = Spectrum1D(flux=temp_cont*u.Jy, spectral_axis=wt1*0.1*u.nm)
        aux2_tmp1 = spline3(aux_tmp1, new_disp_grid*0.1*u.nm)
        matrix_tmp[j]=splineclean(aux2_tmp1.flux.value*fmask)
        tt=np.mean(matrix_tmp[j]**2)
        tb=np.mean((matrix_tmp[j]*new_flux))
        cc=tb/(np.sqrt(norma1)*np.sqrt(tt))
        sigma[j]=cc
        bar1.next()
    bar1.finish()
    print('')
    fig=plt.figure()
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.xlabel("Temp. [1000 K]", fontsize=9)
    plt.ylabel("CCF", fontsize=9)
    plt.plot(vector_t,sigma, color='blue',marker='',ls='-')
    plt.tight_layout()
    # buscar el indice correspondiente al maximo
    jt1=np.argmax(sigma)
    print('\t· · · · · · · · · · · · · · · · · · ·')
    print('\t  Best effective temperature '+str(int(vector_t[jt1]))+' K')
    print('\t· · · · · · · · · · · · · · · · · · ·')
    wt2,ft2 = pyasl.read1dFitsSpec(ltemp[jt1])
    cont2 = continuum(wt2, ft2, order=order, nit=10, type='fit', lo=lo,hi=hi, graph=False)
    spdif = ft2 - cont2
    aux_tmp2 = Spectrum1D(flux=cont2*u.Jy, spectral_axis=wt2*0.1*u.nm)
    aux2_tmp2 = spline3(aux_tmp2, new_disp_grid*0.1*u.nm)
    ajust_cont=splineclean(aux2_tmp2.flux.value)
    # calcular cociente de normas
    # si se define snr se utiliza sigma correspondiente
    if snr==None:
        scalex = np.sqrt(np.mean(spdif**2))/np.sqrt(np.mean(spec_cont**2))
    elif snr > 0:
        ruido = np.sqrt(np.mean(spec_cont**2))/snr
        scalex = np.sqrt(np.mean(spdif**2))/(np.sqrt(np.mean(spec_cont**2))-ruido)
    img_lines = splineclean(aux2_img.flux.value)
    fout = img_lines*scalex + ajust_cont
    fig=plt.figure(figsize=[20,10])
    plt.plot(new_disp_grid,fout,color='red',ls='-',marker='')
    plt.plot(wt2,ft2,color='blue',ls='--',marker='')
    plt.legend((img,'T = '+str(int(vector_t[jt1]))+' K' ))
    plt.tight_layout()





def listmp(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]
    
    
    
