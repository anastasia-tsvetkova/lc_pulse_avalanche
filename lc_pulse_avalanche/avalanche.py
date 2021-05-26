import inspect
import math
from math import exp, log

import matplotlib.pyplot as plt
import numba
import numba as nb
import numpy as np
from numpy.random import exponential, lognormal, normal, uniform
from scipy.stats import loguniform

_EXPANSION_CONSTANT_ = 1.7



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
    
    """
    
    def __init__(self, mu=1.2, mu0=1, alpha=4, delta1=-0.5, delta2=0, tau_min=0.2, tau_max=26,
                 t_min=-10, t_max=1000, res=0.256, eff_area=3600, bg_level=10.67, min_photon_rate=1.3,
                 max_photon_rate=1300, verbose=False):
        
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
                self._rec_gen_pulse(tau, delta_t)
                
        return self._rates
        

    def generate_avalanche_fast(self):


        self._lc_params = []
        
        norms, t_delays, taus, tau_rs = gen_avalanche(mu_0= self._mu0, tau_min= self._tau_min , tau_max= self._tau_max,
                             alpha= self._alpha, resoltuion= self._res, delta1= self._delta1, delta2= self._delta2, mu= self._mu)


        for i,j,k,l in zip(norms, t_delays, taus, tau_rs):
         
            self._lc_params.append(dict(norm=i, t_delay=j, tau=k, tau_r=l))


        return self._lc_params

         
    def generate_avalanche(self):
        """
        Generates pulse avalanche
        
        :returns: set of parameters for the generated avalanche
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
        
        cnt_flux_low = self._min_photon_rate * self._eff_area / self._raw_lc.max()
        cnt_flux_high = self._max_photon_rate * self._eff_area / self._raw_lc.max()
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

    
    def plot_lc(self, rescale=True, save=True, name="./plot_lc.pdf", show_duration=False):
        """
        Plots GRB light curve
        
        :rescale: to rescale the x-axis plotting only lc around T100
        :save: to save the plot to file
        :name: filename (including path) to save the plot
        """
        
        plt.xlabel('T-T0 (s)')
        plt.ylabel('Count rate (cnt/s)')
                
        plt.step(self._times, self._plot_lc)
        
        if rescale:
            t_i = max(self._t_start - 0.3*self._t100, self._t_min)
            t_f = self._t_stop + 0.3*self._t100
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
        
        self._aux_times = self._times[self._raw_lc>self._raw_lc.max()*1e-4]
        self._aux_lc = self._plot_lc[self._raw_lc>self._raw_lc.max()*1e-4]
        
        self._t_start = self._aux_times[0]
        self._t_stop = self._aux_times[-1]
        self._t100 = self._t_stop - self._t_start
        self._total_cnts = np.sum(self._aux_lc) * self._res
                                     
        sum_cnt = 0
        i = 0
        while sum_cnt < 0.05 * self._total_cnts:
            sum_cnt += (self._aux_lc[i] * self._res)
            i += 1
        self._t90_i = self._aux_times[i]
                                     
        sum_cnt = 0
        j = -1
        while sum_cnt < 0.05 * self._total_cnts:
            sum_cnt += (self._aux_lc[j] * self._res)
            j -= 1
        self._t90_f = self._aux_times[j]                             
        
        self._t90 = self._t90_f - self._t90_i            
        self._t90_cnts = np.sum(self._aux_lc[i:j+1]) * self._t90
        
       
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

    
class Restored_LC(LC):
    """
    Class to restore an avalanche from yaml file
    
    :res: GRB LC time resolution
    """
    
    def __init__(self, par_list, res=0.256):
        
        super(Restored_LC, self).__init__(res=res)

        if not par_list:
            raise TypeError("Avalanche parameters should be given")
        elif not isinstance(par_list, list):
            raise TypeError("The avalanche parameters should be a list of dictionaries")
        else:
            self._par_list = par_list
            
        self._raw_lc = np.zeros(len(self._times))
        
        self._restore_lc()

        
    def _restore_lc(self):
        """Restores GRB LC from avalanche parameters"""
        
        for par in self._par_list:
            norm = par['norm']
            t_delay = par['t_delay']
            tau = par['tau']
            tau_r = par['tau_r']
            self._raw_lc += self.norris_pulse(norm, t_delay, tau, tau_r)

        self._plot_lc =  self._raw_lc * self._eff_area + np.random.default_rng().poisson((self._bg), self._n)
        self._aux_lc = self._plot_lc[self._raw_lc>self._raw_lc.max()*1e-4]

        self._get_lc_properties()


def Vector(numba_type):
    """Generates an instance of a dynamically resized vector numba jitclass."""

    if numba_type in Vector._saved_type:
        return Vector._saved_type[numba_type]

    class _Vector:
        """Dynamically sized arrays in nopython mode."""

        def __init__(self, n):
            """Initialize with space enough to hold n garbage values."""
            self.n = n
            self.m = n
            self.full_arr = np.empty(self.n, dtype=numba_type)

        @property
        def size(self):
            """The number of valid values."""
            return self.n

        @property
        def arr(self):
            """Return the subarray."""
            return self.full_arr[: self.n]

        @property
        def last(self):
            """The last element in the array."""
            if self.n:
                return self.full_arr[self.n - 1]
            else:
                raise IndexError("This numbavec has no elements: cannot return 'last'.")

        @property
        def first(self):
            """The first element in the array."""
            if self.n:
                return self.full_arr[0]
            else:
                raise IndexError(
                    "This numbavec has no elements: cannot return 'first'."
                )

        def clear(self):
            """Remove all elements from the array."""
            self.n = 0
            return self

        def extend(self, other):
            """Add the contents of a numpy array to the end of this Vector.

            Arguments
            ---------
            other : 1d array
                The values to add to the end.
            """
            n_required = self.size + other.size
            self.reserve(n_required)
            self.full_arr[self.size : n_required] = other
            self.n = n_required
            return self

        def append(self, val):
            """Add a value to the end of the Vector, expanding it if necessary."""
            if self.n == self.m:
                self._expand()
            self.full_arr[self.n] = val
            self.n += 1
            return self

        def reserve(self, n):
            """Reserve a n elements in the underlying array.

            Arguments
            ---------
            n : int
                The number of elements to reserve

            Reserving n elements ensures no resize overhead when appending up
            to size n-1 .
            """
            if n > self.m:  # Only change size if we are
                temp = np.empty(int(n), dtype=numba_type)
                temp[: self.n] = self.arr
                self.full_arr = temp
                self.m = n
            return self

        def consolidate(self):
            """Remove unused memory from the array."""
            if self.n < self.m:
                self.full_arr = self.arr.copy()
                self.m = self.n
            return self

        def __array__(self):
            """Array inteface for Numpy compatibility."""
            return self.full_arr[: self.n]

        def _expand(self):
            """Internal function that handles the resizing of the array."""
            self.m = int(self.m * _EXPANSION_CONSTANT_) + 1
            temp = np.empty(self.m, dtype=numba_type)
            temp[: self.n] = self.full_arr[: self.n]
            self.full_arr = temp

        def set_to(self, arr):
            """Make this vector point to another array of values.

            Arguments
            ---------
            arr : 1d array
                Array to set this vector to. After this operation, self.arr
                will be equal to arr. The dtype of this array must be the 
                same dtype as used to create the vector. Cannot be a readonly
                vector.
            """
            self.full_arr = arr
            self.n = self.m = arr.size

        def set_to_copy(self, arr):
            """Set this vector to an array, copying the underlying input.

            Arguments
            ---------
            arr : 1d array
                Array to set this vector to. After this operation, self.arr
                will be equal to arr. The dtype of this array must be the 
                same dtype as used to create the vector.
            """
            self.full_arr = arr.copy()
            self.n = self.m = arr.size

    if numba_type not in Vector._saved_type:
        spec = [("n", numba.uint64), ("m", numba.uint64), ("full_arr", numba_type[:])]
        Vector._saved_type[numba_type] = numba.experimental.jitclass(spec)(_Vector)

    return Vector._saved_type[numba_type]


Vector._saved_type = dict()

VectorUint8 = Vector(numba.uint8)
VectorUint16 = Vector(numba.uint16)
VectorUint32 = Vector(numba.uint32)
VectorUint64 = Vector(numba.uint64)

VectorInt8 = Vector(numba.int8)
VectorInt16 = Vector(numba.int16)
VectorInt32 = Vector(numba.int32)
VectorInt64 = Vector(numba.int64)

VectorFloat32 = Vector(numba.float32)
VectorFloat64 = Vector(numba.float64)

VectorComplex64 = Vector(numba.complex64)
VectorComplex128 = Vector(numba.complex128)

__all_types = tuple(v for k, v in Vector._saved_type.items())


def _isinstance(obj):
    return isinstance(obj, __all_types)


@nb.njit(fastmath=True)
def gen_avalanche(mu_0, tau_min, tau_max, alpha, resoltuion, delta1, delta2, mu):

    # number of spontaneous primary pulses: p5(mu_s) = exp(-mu_s/mu0)/mu0
    mu_s = np.round(np.random.exponential(scale=mu_0))
    if mu_s == 0:
        mu_s = 1 

    norms = VectorFloat64(0)
    t_delays = VectorFloat64(0)
    taus = VectorFloat64(0)
    tau_rs = VectorFloat64(0)


    log_tau_min = np.log(tau_min)
    log_tau_max = np.log(tau_max)

    
    
    for i in range(mu_s):
        # time constant of spontaneous pulses: p6(log tau0) = 1/(log tau_max - log tau_min)
        # decay time

        tau0 = math.exp(np.random.uniform(log_tau_max, log_tau_min))
    
        # rise time
        tau_r = 0.5 * tau0

        # time delay of spontaneous primary pulses: p7(t) = exp(-t/(alpha*tau0))/(alpha*tau0)
        t_delay = np.random.exponential(scale = alpha*tau0)

        # pulse amplitude: p1(A) = 1 in [0, 1]
        norm = np.random.rand()

        norms.append(norm)
        t_delays.append(t_delay)
        taus.append(tau0)
        tau_rs.append(tau_r)

        
        tau1 = tau_r

        t_shift = t_delay

        while tau1 > resoltuion:

            mu_b = np.round(np.random.exponential(scale= mu))
            
            for i in range(mu_b):

                # time const of the baby pulse: p4(tau/tau1) = 1/(delta2 - delta1), tau1 - time const of the parent pulse
                tau = tau1 * math.exp(np.random.uniform(delta1, delta2))

                tau_r = 0.5 * tau

                # time delay of baby pulse: p3(delta_t) = exp(-delta_t/(alpha*tau))/(alpha*tau) with respect to the parent pulse, 
                # alpha - delay parameter, tau - time const of the baby pulse
                delta_t = np.random.exponential(scale=alpha*tau) + t_shift

                norm = np.random.rand()
                
                norms.append(norm)
                t_delays.append(delta_t)
                taus.append(tau)
                tau_rs.append(tau_r)

                tau1 = tau_r

                t_shift = delta_t
                
    return norms.arr, t_delays.arr, taus.arr, tau_rs.arr
    
