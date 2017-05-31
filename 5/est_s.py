from matplotlib import pyplot, mlab
import numpy
from fftfilt import fftfilt

def est_s(s,r,nfft=2**12):
    '''
    Q_rs = numpy.correlate(r_est, s,'full')
    Q_rr = numpy.correlate(r_est, r_est,'full')
    Qf_rs = numpy.fft.fftpack.rfft(Q_rs,nfft)
    Qf_rr = numpy.fft.fftpack.rfft(Q_rr,nfft)
    '''
    Qf_rs = mlab.csd(r,s,nfft) 
    Qf_rr = mlab.csd(r,r,nfft) 

    Txy = Qf_rs[0]/Qf_rr[0]
    T= numpy.array(Txy.tolist() + Txy[::-1].conj()[1:-1].tolist())
    
    k = numpy.fft.fftshift(numpy.fft.ifft(T).real)

    '''
    s_est = fftfilt(k,r)
    s_est = s_est[nfft/2:]
    '''
    s_est = numpy.convolve(r,k,'full')
    s_est = s_est[nfft/2:-nfft/2+1]

    return s_est, Txy, T, k
