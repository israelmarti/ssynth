<h1 align="center"> ssynth </h1>

 ## Introduction


**ssynth** is a PYTHON library applied to generate synthetic spectroscopic observational datasets of a simulated binary system.


## Installation

**ssynth** is distributed on PyPI as a universal wheel is available on Linux/macOS and Windows and supports Python 3.10.12

```bash
pip install git+https://github.com/israelmarti/ssynth#egg=ssynth
```
For the optimized interactive useful run from [IPython](https://ipython.org/install.html).

## Dependencies

**ssynth** requires the latest dependencies for Python 3.10 (for 3.6 and 3.8 install these versions):
- [Astropy](https://www.astropy.org) (v6.0.0)
- [Matplotlib](https://matplotlib.org) (v3.8.3)
- [Numba](https://numba.pydata.org) (v0.59.0)
- [Numpy](https://www.numpy.org)  (v1.26.4)
- [Progress](https://pypi.org/project/progress) (v1.6)
- [PyAstronomy](https://pyastronomy.readthedocs.io) (v0.20.0)
- [SciPy](https://scipy.org) (v1.12.0)
- [Specutils](https://specutils.readthedocs.io) (v1.13.0)

## Main Functions

- **create**

> Build a dataset from any two component spectra and orbital parameters provided by the user. The variables are: number of observations ($N$), flux ratio ($L_B/L_A$), mass ratio ($M_B/M_A$), orbital period ($P$), span time, tilt angle ($i$), RV measurement random error, and S/N (as Gaussian noise added to the composite spectrum). 
> 
> Mandatory parameters:
> - `sa`: spectrum of primary component in FITS extension with or without extension (string);
> - `sb`: spectrum of secondary component in FITS extension with or without extension (string);
> - `ma`: mass of primary component (float);
> - `mb`: mass of secondary component (float);
> - `la`: luminosity of primary component (float);
> - `lb`: luminosity of secondary component (float);
> - `n`: number of observations  (integer);
> - `p`: orbital period of binary system in days (float);
> - `yran`: time span in years (float);
> 
> Optional parameters:
> - `i`: tilt angle (float);
> - `err`: RV measurement random error (float);
> - `snr`: Gaussian noise added to the composite spectrum (float);
> - `vgamma`: estimated value for systemic radial velocity in km/s (float);   
> - `T0`: epoch zero (float);
> - `obj`: keyword object name (string).
> 
> Example:
```python3
create('BD-11467.fits', 'GJ1012.fits', 0.63, 0.22, 0.55, 0.003, 25, 800, 5.5, i=83, err=0.5, snr=150, vgamma=-2.23, T0=2451884.5, obj='example')
```

- **fxcor**
> Compute radial velocities via Fourier cross correlation (FCC) between one dimension spectrum with a template (FITS format).
> 
> Mandatory parameters:
> - `img`: observed spectrum in FITS extension with or without extension (string);
> - `tmp`: template spectrum in FITS extension with or without extension (string);
> 
> Optional parameters:
> - `rvrange`: RV range cosidered for FCC, and expressed in km/s (float);
> - `order`: Chebyshev polynomial order (integer);
> - `wreg`: spectral regions for cross-correlation analysis (string). The selected region is specified among "-" and the different regions joined with ' " , ";
> 
> Example:
```python3
fxcor('BD-11467.fits', 'template_A.fits',rvrange=1000,order=10,wreg=wreg='3800-4700,4900-5800')
```

- **fxcompare**
> Compute radial velocities via Fourier cross correlation (FCC) between two dimension specta with a template (FITS format).
> 
> Mandatory parameters:
> - `img1`: first observed spectrum in FITS extension with or without extension (string);
> - `img2`: second observed spectrum in FITS extension with or without extension (string);
> - `tmp`: template spectrum in FITS extension with or without extension (string);
> 
> Optional parameters:
> - `rvrange`: RV range cosidered for FCC, and expressed in km/s (float);
> - `order`: Chebyshev polynomial order (integer);
> - `ystep`: flux step scale (float);
> - `wreg`: spectral regions for cross-correlation analysis (string). The selected region is specified among "-" and the different regions joined with ' " , ";
> 
> Example:
```python3
fxcor('BD-11467_A.fits', 'BD-11467_B.fits', 'template_A.fits',rvrange=1000,order=10,wreg=wreg='3800-4700,4900-5800')
```

- **continuum**
> Continuum normalize spectra (array format inputs).
> 
> Mandatory parameters:
> - `w`: array dispersion grid (array).
> - `f`: flux values (array).
> 
> Optional parameters:
> - `order`: Chebyshev polynomial order (integer);
> - `type`: type of output spectra.  The choices are "fit"  for  the  fitted function,  "ratio"  for  the  ratio  of the input spectra to the fit, "diff" for the difference between the  input  spectra  and  the  fit (string).
> - `lo`: rejection limits below the  fit  in  units  of  the residual sigma;
> - `hi`: rejection limits above the  fit  in  units  of  the residual sigma (float);
> - `nit`: number of rejection iterations (integer);
> - `graph`: show continuum fitting (boolean).
> 
> Example:
```python3
continuum(wgrid, flux, order=12, type='fit', lo=2, hi=3, nit=10, graph=True):
```
