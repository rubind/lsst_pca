import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
import sncosmo
from matplotlib.backends.backend_pdf import PdfPages


def Kim13_PCs():
    dat = np.loadtxt("input/LC_PCs.txt")
    print dat
    bands = np.array(["griz"[int(item)] for item in dat[:,0]])
    pcs = dat[:,1]
    phases = dat[:,2]
    mags = dat[:,3]

    interpfns = {}
    for i, band in enumerate('griz'):
        #plt.subplot(2,2,i+1)
        #plt.title(band)
        for pc in range(4):
            inds = np.where((bands == band)*(pcs == pc))
            phase = phases[inds]
            mag = mags[inds]
            
            phase = np.concatenate(([-100.], phase, [100.]))
            mag = np.concatenate(([mag[0]], mag, [mag[-1]]))
            
            interpfns[(band, pc)] = interp1d(phase, mag, kind = 'linear')
            #plt.plot(np.arange(-10., 36.), interpfns[(band, pc)](np.arange(-10., 36.)), label = str(pc))
        #plt.legend(loc = 'best')
    #plt.savefig("pc_interps.pdf")
    #plt.close()
    return interpfns



def LC_eval(interpfns, phase, magerr, filt, redshift):
    obs_to_pc_filt = {"r": "g", "i": "r", "z": "i", "Y": "z"}

    jacobian = np.zeros([len(phase), 5], dtype=np.float64) # parameters are daymax, mag, pc0123
    jacobian[:,0] = 1.

    weight_matrix = np.zeros([len(phase)]*2, dtype=np.float64)
    total_SNR_all = 0.

    for i in range(len(phase)):
        if "rizY".count(filt[i]) and magerr[i] > 0 and magerr[i] < 0.5:
            weight_matrix[i, i] = 1./magerr[i]**2.

            for k in range(4):
                jacobian[i, k+1] = interpfns[(obs_to_pc_filt[filt[i]], k)](phase[i])
        else:
            weight_matrix[i,i] = 0.0001

    param_wmat = np.dot(np.transpose(jacobian), np.dot(weight_matrix, jacobian))
    param_cmat = np.linalg.inv(param_wmat)
    return np.sqrt(np.diag(param_cmat))
    

def read_and_eval(interpfns):
    f = fits.open("input/LSST_Ia_HEAD.FITS")
    head = f[1].data
    f.close()

    dat = sncosmo.read_snana_fits("input/LSST_Ia_HEAD.FITS", "input/LSST_Ia_PHOT.FITS")
    pdf = PdfPages("LC_plots.pdf")

    for i in range(len(head.PEAKMJD)):
        if (head.REDSHIFT_HELIO[i] > 0.4) and (head.REDSHIFT_HELIO[i] < 0.6):
            plt.figure()

            phase = (dat[i]["MJD"] - head.PEAKMJD[i])/(1. + head.REDSHIFT_HELIO[i])
            
            errs = LC_eval(interpfns, phase, magerr = dat[i]["MAGERR"], filt = dat[i]["FLT"], redshift = head.REDSHIFT_HELIO[i])
            
            for filt in 'ugrizY':
                inds = np.where((dat[i]["FLT"] == filt)*(dat[i]["MAGERR"] > 0))
                plt.plot(phase[inds], dat[i]["MAGERR"][inds], '.', color = {'u': 'm', 'g': 'b', 'r': 'cyan', 'i': 'g', 'z': 'orange', 'Y': 'r'}[filt], label = filt)
            plt.legend(loc = 'best')
            plt.ylim(0, 0.2)
            title = "$\sigma$Mag=%.3f" % errs[0]
            for pc in range(1,5):
                title += " $\sigma$PC%i=%.3f" % (pc, errs[pc])

            plt.title(title)
            plt.xlabel("Phase")
            plt.ylabel("Mag Err")
            pdf.savefig(plt.gcf(), bbox_inches = 'tight')
    pdf.close()

interpfns = Kim13_PCs()
read_and_eval(interpfns)
