import inspect
import math
from math import exp, log
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential, lognormal, normal, uniform
from scipy.stats import loguniform
import os, h5py

class LC(object):
    """
    A class to generate gamma-ray burst light curves (GRB LCs) using a pulse avalance model ('chain reaction') 
    proposed by Stern & Svensson, ApJ, 469: L109 (1996).
    
    :mu: average number of baby pulses
    :mu0: average number of spontaneous (initial) pulses
    :alpha: delay parameter
    :delta1: lower boundary of log-normal probability distribution of tau (time constant of baby pulse)
    :delta2: upper boundary of log-normal probability distribution of tau
    :tau_min: lower boundary of log-normal probability distribution of tau_0 (time constant of spontaneous pulse);
             should be smaller than res
    :tau_max: upper boundary of log-normal probability distribution of tau_0
    :t_min: GRB LC start time
    :t_max: GRB LC stop time
    :res: GRB LC time resolution
    :eff_area: effective area of instrument (cm2)
    :bg_level: backgrounf level (cnt/cm2/s)
    :min_photon_rate: left boundary of -3/2 log N - log S distrubution (ph/cm2/s)
    :max_photon_rate: right boundary of -3/2 log N - log S distrubution (ph/cm2/s)
    :sigma: signal above background level
    :n_cut: maximum number of pulses in avalanche (useful to speed up the simulations but in odds with the "classic" approach)
    """
    
    def __init__(self, mu=1.2, mu0=1, alpha=4, delta1=-0.5, delta2=0, tau_min=0.2, tau_max=26,
                 t_min=-10, t_max=1000, res=0.256, eff_area=3600, bg_level=10.67, min_photon_rate=1.3,
                 max_photon_rate=1300, sigma=5, n_cut=None, verbose=False):
        
        self._mu = mu # mu~1 => critical runaway regime
        self._mu0 = mu0 # average number of spontaneous pulses per GRB
        self._alpha = alpha # delay parameter
        self._delta1 = delta1
        self._delta2 = delta2
        if tau_min > res and not isinstance(self, Restored_LC):
            raise ValueError("tau_min should be smaller than res =", res)
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._eff_area = eff_area 
        self._bg = bg_level * self._eff_area # cnts/s
        self._min_photon_rate = min_photon_rate  
        self._max_photon_rate = max_photon_rate 
        self._verbose = verbose
        self._res = res # s
        self._n = int(np.ceil((t_max - t_min)/self._res)) + 1
        self._t_min = t_min
        self._t_max = (self._n - 1) * self._res + self._t_min
        self._times, self._step = np.linspace(self._t_min, self._t_max, self._n, retstep=True)
        self._rates = np.zeros(len(self._times))
        self._sp_pulse = np.zeros(len(self._times))
        self._total_rates = np.zeros(len(self._times))
        self._lc_params = list()
        self._sigma = sigma
        self._n_cut = n_cut
        self._n_pulses = 0
        
        if self._verbose:
            print("Time resolution: ", self._step)
                
     
    def norris_pulse(self, norm, tp, tau, tau_r):
        """
        Computes a single pulse according to Norris te al., ApJ, 459, 393 (1996)
        
        :t: times (lc x-axis), vector
        :tp: pulse peak time, scalar
        :tau: pulse width (decay rime), scalar
        tau_r: rise time, scalar
        :returns: an array of count rates
        """

        self._n_pulses += 1
        
        if self._verbose:
            print("Generating a new pulse with tau={:0.3f}".format(tau))

        t = self._times 
        _tp = np.ones(len(t))*tp
        
        if tau_r == 0 or tau == 0: 
            return np.zeros(len(t))
        
        return np.append(norm * np.exp(-(t[t<=tp]-_tp[t<=tp])**2/tau_r**2), \
                         norm * np.exp(-(t[t>tp]-_tp[t>tp])/tau))

   
    def _rec_gen_pulse(self, tau1, t_shift):
        """
        Recursively generates pulses from Norris function
        
        :tau1: parent pulse width (decay rime), scalar
        :t_shift: time delay relative to the parent pulse
        
        :returns: an array of count rates
        """
            
        # number of baby pulses: p2(mu_b) = exp(-mu_b/mu)/mu, mu - the average, mu_b - number of baby pulses
        mu_b = round(exponential(scale=self._mu))
                
        if self._verbose:
            print("Number of pulses:", mu_b)
            print("--------------------------------------------------------------------------")
        
        for i in range(mu_b):
           
            # time const of the baby pulse: p4(tau/tau1) = 1/(delta2 - delta1), tau1 - time const of the parent pulse
            tau = tau1 * exp(uniform(low=self._delta1, high=self._delta2))
            
            tau_r = 0.5 * tau
            
            # time delay of baby pulse: p3(delta_t) = exp(-delta_t/(alpha*tau))/(alpha*tau) with respect to the parent pulse, 
            # alpha - delay parameter, tau - time const of the baby pulse
            delta_t = exponential(scale=self._alpha*tau) + t_shift
            
            norm = uniform(low=0.0, high=1)
            
            self._rates += self.norris_pulse(norm, delta_t, tau, tau_r) 
            
            self._lc_params.append(dict(norm=norm, t_delay=delta_t, tau=tau, tau_r=tau_r))
            
            if self._verbose:
                print("Pulse amplitude: {:0.3f}".format(norm))
                print("Pulse shift: {:0.3f}".format(delta_t))
                print("Time constant (the decay time): {0:0.3f}".format(tau))
                print("Rise time: {:0.3f}".format(tau_r))
                print("--------------------------------------------------------------------------")
                
            if tau > self._res:
                if self._n_cut is None:
                    self._rec_gen_pulse(tau, delta_t)
                else:
                    if self._n_pulses < self._n_cut:
                        self._rec_gen_pulse(tau, delta_t)

        return self._rates
        
        
    def generate_avalanche(self, seed=12345, return_array=False):
        """
        Generates pulse avalanche
        
        :seed: random seed
        :return_array: if True returns arrays of parameters, if False - a dict with parameters for each pulse
        :returns: set of parameters for the generated avalanche
        """
        
        # set seed for random draw (the same as for the avalanche generation)
        np.random.seed(seed)
        
        if self._verbose:
            inspect.getdoc(self.generate_avalanche)
            
        """
        Starting pulse avalanche
        """
   
        # number of spontaneous primary pulses: p5(mu_s) = exp(-mu_s/mu0)/mu0
        mu_s = round(exponential(scale=self._mu0))
        if mu_s == 0:  mu_s = 1 
            
        if self._verbose:
            print("Number of spontaneous pulses:", mu_s)
            print("--------------------------------------------------------------------------")
        
        for i in range(mu_s):
            # time constant of spontaneous pulses: p6(log tau0) = 1/(log tau_max - log tau_min)
            # decay time
            tau0 = exp(uniform(low=log(self._tau_max), high=log(self._tau_min)))

            # rise time
            tau_r = 0.5 * tau0

            # time delay of spontaneous primary pulses: p7(t) = exp(-t/(alpha*tau0))/(alpha*tau0)
            t_delay = exponential(scale=self._alpha*tau0)

            # pulse amplitude: p1(A) = 1 in [0, 1]
            norm = uniform(low=0.0, high=1) 
                     
            if self._verbose:
                print("Spontaneous pulse amplitude: {:0.3f}".format(norm))
                print("Spontaneous pulse shift: {:0.3f}".format(t_delay))
                print("Time constant (the decay time) of spontaneous pulse: {0:0.3f}".format(tau0))
                print("Rise time of spontaneous pulse: {:0.3f}".format(tau_r))
                print("--------------------------------------------------------------------------")
                
            self._sp_pulse += self.norris_pulse(norm, t_delay, tau0, tau_r)
            
            self._lc_params.append(dict(norm=norm, t_delay=t_delay, tau=tau0, tau_r=tau_r))
            
            self._rec_gen_pulse(tau0, t_delay)
        
        # lc directly from the avalanche
        self._raw_lc = self._sp_pulse + self._rates

        self._max_raw_pcr = self._raw_lc.max()
        population = np.geomspace(self._min_photon_rate , self._max_photon_rate, 1000)
        weights = list(map(lambda x: x**(-3/2), population))
        weights = weights / np.sum(weights)
        ampl = np.random.choice(population, p=weights) / self._max_raw_pcr
        
        self._peak_value = self._max_raw_pcr * ampl
        
#         lc from avalanche scaled + Poissonian bg added
        self._plot_lc = self._raw_lc * ampl * self._eff_area + np.random.default_rng().poisson((self._bg), self._n)

        self._get_lc_properties()
        
        for p in self._lc_params:
            p['norm'] *= ampl
        
        norms = np.empty((0,))
        t_delays = np.empty((0,))
        taus = np.empty((0,))
        tau_rs = np.empty((0,))
        
        if return_array:
            for p in self._lc_params:
                norms = np.append(norms, p['norm'])
                t_delays = np.append(t_delays, p['t_delay'])
                taus = np.append(taus, p['tau'])
                tau_rs = np.append(tau_rs, p['tau_r'])
                
            return norms, t_delays, taus, tau_rs, self._peak_value
        
        else:
            return self._lc_params

   
    def plot_lc(self, rescale=True, save=True, name="./plot_lc.pdf", show_duration=False):
        """
        Plots GRB light curve
        
        :rescale: to rescale the x-axis plotting only lc around T100
        :save: to save the plot to file
        :name: filename (including path) to save the plot
        """
        
        plt.xlabel('T-T0 (s)')
        plt.ylabel('Count rate (cnt/s)')
                
        self._restore_lc()
        
        plt.step(self._times, self._plot_lc, where='post')
        plt.plot(np.linspace(self._t_min, self._t_max, num=2, endpoint=True), [self._bg, self._bg], 'r--')
        
        if rescale:
            t_i = max(self._t_start - 0.5*self._t100, self._t_min)
            t_f = self._t_stop + 0.5*self._t100
            plt.xlim([t_i, t_f])
            
        if show_duration:
                plt.axvline(x=self._t_start, color='blue')
                plt.axvline(x=self._t_stop, color='blue')
                plt.axvline(x=self._t90_i, color='red')
                plt.axvline(x=self._t90_f, color='red')
          
        if save:
            plt.savefig(name)
        
        plt.show()
            
    
    def _get_lc_properties(self):
        """
        Calculates T90 and T100 durations along with their start and stop times, total number of counts per T100, 
        mean, max, and background count rates
        """
        
        self._aux_index = np.where(self._raw_lc>self._raw_lc.max()*1e-4)
#         self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
        self._max_snr = ((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5).max()
        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
        self._aux_lc = self._plot_lc[self._aux_index[0][0]:self._aux_index[0][-1]]

        self._t_start = self._times[self._aux_index[0][0]]
#         self._t_stop = self._times[self._aux_index[0][-1]+1]
        self._t_stop = self._times[self._aux_index[0][-1]]
            
        self._t100 = self._t_stop - self._t_start
        
        self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
                
        try:
            sum_cnt = 0
            i = 0
            while sum_cnt < 0.05 * self._total_cnts:
                sum_cnt += (self._aux_lc[i] - self._bg) * self._res
                i += 1
                
            self._t90_i = self._aux_times[i]
                                     
            sum_cnt = 0
            j = -1
            while sum_cnt < 0.05 * self._total_cnts:
                sum_cnt += (self._aux_lc[j] - self._bg) * self._res
                j += -1

            self._t90_f = self._aux_times[j]      

            self._t90 = self._t90_f - self._t90_i            
            self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
            
            assert self._t90_i < self._t90_f
            
        except:
            self._t90 = self._t100
            self._t90_i = self._t_start
            self._t90_f = self._t_stop
            self._t90_cnts = self._total_cnts
           

    @property
    def T90(self):
        return "{:0.3f}".format(self._t90), "{:0.3f}".format(self._t90_i), "{:0.3f}".format(self._t90_f)
    
    @property
    def T100(self):
        return "{:0.3f}".format(self._t100), "{:0.3f}".format(self._t_start), "{:0.3f}".format(self._t_stop)
    
    @property
    def total_counts(self):
        return "{:0.2f}".format(self._total_cnts)
    
    @property
    def max_rate(self):
        return "{:0.2f}".format(self._aux_lc.max())
    
    @property
    def mean_rate(self):
        return "{:0.2f}".format(np.mean(self._aux_lc))
    
    @property
    def bg_rate(self):
        return "{:0.2f}".format(self._bg)
    
    @property
    def max_snr(self):
        return "{:0.2f}".format(self._max_snr)
    
    
    def _restore_lc(self):
        """Restores GRB LC from avalanche parameters"""
        
        self._raw_lc = np.zeros(len(self._times))
        
        for par in self._lc_params:
            norm = par['norm']
            t_delay = par['t_delay']
            tau = par['tau']
            tau_r = par['tau_r']
            self._raw_lc += self.norris_pulse(norm, t_delay, tau, tau_r) #

        self._plot_lc =  self._raw_lc * self._eff_area + np.random.default_rng().poisson((self._bg), self._n)

        self._get_lc_properties()
        
        
    def hdf5_lc_generation(self, outfile, overwrite=False, seed=12345):
        """
        Generates a new avalanche and writes it to an hdf5 file
        
        :n_lcs: number of light curves we want to simulate
        :outfile: file name
        :overwrite: overwrite existing file
        :seed: random seed for the avalanche generation, int or list
        """
        
        if overwrite == False:
            assert os.path.isfile(outfile), 'ERROR: file already exists!'

        self._f = h5py.File(outfile, 'w')

        
        self._f.create_group('GRB_PARAMETERS')
        self._f['GRB_PARAMETERS'].attrs['PARAMETER_ORDER'] = '[K, t_start, t_rise, t_decay]'

        self._grb_counter = 1
            
        if isinstance(seed, list):
            for sd in seed:
                self.aux_hdf5(seed=sd)
                
        else:
            self.aux_hdf5(seed=seed)

        self._f.close()
        
        
    def aux_hdf5(self, seed):
        norms, t_delays, taus, tau_rs, peak_value = self.generate_avalanche(seed=seed, return_array=True)
        n_pulses = norms.size

        grb_array = np.concatenate((
                    norms.reshape(n_pulses,1),
                    t_delays.reshape(n_pulses,1),
                    tau_rs.reshape(n_pulses,1),
                    taus.reshape(n_pulses,1)),
                    axis=1
                    )

        self._f.create_dataset(f'GRB_PARAMETERS/GRB_{self._grb_counter}', data=grb_array)
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['PEAK_VALUE'] = peak_value
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['N_PULSES'] = n_pulses
        self._grb_counter += 1

    
class Restored_LC(LC):
    """
    Class to restore an avalanche from yaml file
    
    :res: GRB LC time resolution
    """
    
    def __init__(self, par_list, res=0.256, t_min=-10, t_max=1000, sigma=5):
        
        super(Restored_LC, self).__init__(res=res, t_min=t_min, t_max=t_max, sigma=sigma)

        if not par_list:
            raise TypeError("Avalanche parameters should be given")
        elif not isinstance(par_list, list):
            raise TypeError("The avalanche parameters should be a list of dictionaries")
        else:
            self._lc_params = par_list
            
 
        self._restore_lc()
