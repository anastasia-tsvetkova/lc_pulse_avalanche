# Stern & Svensson, ApJ, 469: L109 (1996)

from math import exp, log
import inspect
from scipy.stats import loguniform
from numpy.random import exponential, uniform, lognormal, normal
import numpy as np
import matplotlib.pyplot as plt


class LC(object):
    
    def __init__(self, mu=1.2, mu0=1, alpha=4, delta1=-0.5, delta2=0, tau_min=0.2, tau_max=26, \
                 t_min=-10, t_max=1000, res=0.256, eff_area=3600, bg_level=10.67, verbose=False):
        
        self._mu = mu # mu~1 => critical runaway regime
        self._mu0 = mu0 # average number of spontaneous pulses per GRB
        self._alpha = alpha # delay parameter
        self._delta1 = delta1
        self._delta2 = delta2
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._eff_area = eff_area
        self._bg = bg_level * self._eff_area # cnts/s
        self._photon_rate = 0.65 # ph/cm2/s
        self._verbose = verbose
        self._res = res
        self._n = int(np.ceil((t_max - t_min)/self._res)) + 1
        self._t_min = t_min
        self._t_max = (self._n - 1) * self._res + self._t_min
        self._times, self._step = np.linspace(t_min, self._t_max, self._n, retstep=True)
        self._rates = np.zeros(len(self._times))
        self._sp_pulse = np.zeros(len(self._times))
        self._total_rates = np.zeros(len(self._times))
        self._lc_params = list()
        
        if self._verbose:
            print("Time resolution: ", self._step)
                
        
    def norris_pulse(self, norm, tp, tau, tau_r):
        """
        Norris te al., ApJ, 459, 393 (1996)
        :t: times (lc x-axis), vector
        :tp: pulse peak time, scalar
        :tau: pulse width (decay rime), scalar
        tau_r: rise time, scalar
        :returns: an array of count rates
        """

        if self._verbose:
            print("Generating a new pulse with tau={:0.3f}".format(tau))

        t = self._times 
        _tp = np.ones(len(t))*tp
        return np.append(norm * np.exp(-(t[t<=tp]-_tp[t<=tp])**2/tau_r**2), \
                         norm * np.exp(-(t[t>tp]-_tp[t>tp])/tau))
    
    
    def _rec_gen_pulse(self, tau1, t_shift):
        """
        Recursively generating pulses from Norris function
        
        :tau1: parent pulse width (decay rime), scalar
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
                self._rec_gen_pulse(tau, delta_t)
                
        return self._rates
        
    
    def generate_avalanche(self):
        """
        Generating pulse avalanche
        """
        
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
        
        cnt_flux_low = self._photon_rate * self._eff_area / self._raw_lc.max()
        cnt_flux_high = 2e3 * cnt_flux_low
        population = np.geomspace(cnt_flux_low, cnt_flux_high, 1000)
        weights = list(map(lambda x: x**(-3/2), population))
        weights = weights / np.sum(weights)
        ampl = np.random.choice(population, p=weights)
        
        # lc from avalanche scaled + Poissonian bg added
        self._plot_lc = self._raw_lc * ampl + np.random.default_rng().poisson((self._bg), self._n)
        
        self._get_lc_properties()
        
        for p in self._lc_params:
            p['norm'] *= ampl/self._eff_area 

        return self._lc_params

    
    def plot_lc(self, rescale=True, save=True):
        """
        Plotting light curve
        
        :rescale: to rescale the x-axis plotting only lc around T100, bool
        """
        plt.xlabel('T-T0 (s)')
        plt.ylabel('Count rate (cnt/s)')

        self._plt = plt.step(self._times, self._plot_lc)
        
        if rescale:
            t_i = max(self._t_start - 0.3*self._t100, self._t_min)
            t_f = self._t_stop + 0.3*self._t100
            plt.xlim([t_i, t_f])
            
        if save:
            plt.savefig("/home/anastasia/work/Galileo/LCs/fig_report.pdf")
    
    def _get_lc_properties(self):
        """
        Calculaing T90 duration
        """
        
        self._aux_times = self._times[self._raw_lc>self._raw_lc.max()*1e-4]
        self._aux_lc = self._plot_lc[self._raw_lc>self._raw_lc.max()*1e-4]
        
        self._t_start = self._aux_times[0]
        self._t_stop = self._aux_times[-1]
        self._t100 = self._t_stop - self._t_start
        self._total_cnts = np.sum(self._aux_lc)
                                     
        sum_cnt = 0
        i = 0
        while sum_cnt < 0.05 * self._total_cnts:
            sum_cnt += self._aux_lc[i]                
            i += 1
        self._t90_i = self._aux_times[i]
                                     
        sum_cnt = 0
        j = -1
        while sum_cnt < 0.05 * self._total_cnts:
            sum_cnt += self._aux_lc[j]                
            j -= 1
        self._t90_f = self._aux_times[j]                             
        
        self._t90 = self._t90_f - self._t90_i            
        self._t90_cnts = np.sum(self._aux_lc[i:j+1])
        
       
    @property
    def T90(self):
        return "{:0.3f}".format(self._t90), "{:0.3f}".format(self._t90_i), "{:0.3f}".format(self._t90_f)
    
    @property
    def T100(self):
        return "{:0.3f}".format(self._t100), "{:0.3f}".format(self._t_start), "{:0.3f}".format(self._t_stop)
    
    @property
    def total_rate(self):
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

    
class Restored_LC(LC):
    """
    Class to restore an avalanche from yaml file
    """
    
    def __init__(self, par_list):
        
        super(Restored_LC, self).__init__(t_min=-10, t_max=1000, res=0.256, eff_area=3600, \
                                          bg_level=10.67, verbose=False)

        if not par_list:
            raise TypeError("Avalanche parameters should be given")
        elif not isinstance(par_list, list):
            raise TypeError("The avalanche parameters should be a list of dictionaries")
        else:
            self._par_list = par_list
            
        self._raw_lc = np.zeros(len(self._times))
    
    def restore_lc(self):
        
        for par in self._par_list:
            norm = par['norm']
            t_delay = par['t_delay']
            tau = par['tau']
            tau_r = par['tau_r']
            self._raw_lc += self.norris_pulse(norm, t_delay, tau, tau_r)

        self._plot_lc =  self._raw_lc * self._eff_area + np.random.default_rng().poisson((self._bg), self._n)
        self._aux_lc = self._plot_lc[self._raw_lc>self._raw_lc.max()*1e-4]
