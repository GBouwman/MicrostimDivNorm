import numpy as np
import math
from scipy.stats import gamma

class Adapted_DN_Model:
    """ Adaptation of the Delayed Divisive Normalization model by Brands et al. https://doi.org/10.1371/journal.pcbi.1012161,
    based on code by Brands et al.

    Simulation of several temporal models to predict a neuronal response given a stimulus
    time series as input.

    This class contains the following modeling components:
    lin : convolves input with an Impulse Response Function (IRF)
    rectf : full-wave rectification
    exp : exponentiation

    Options for divise normalization (i.e. computation for the value of the denominator):
    norm : normalization of the input with a semi-saturation constant
    delay: delayed normalization of the input with a semi-saturation constant

    params
    -----------------------
    stim : array dim(T)
        stimulus time course
    sample_rate : int
        frequency with which the timepoints are measured
    adapt : float
        controls adaptation of stimuli
    tau : float
        time to peak for positive IRF (seconds)
    weight : float
        ratio of negative to positive IRFs
    shift : float
        time between stimulus onset and when the signal reaches the cortex (seconds)
    scale : float
        response gain
    n : float
        exponent
    sigma : float
        semi-saturation constant of fast adaptation
    tau_b : float
        time window of fast adaptation (seconds)
    sigma_a : float
        semi-saturation constant of slow adaptation
    tau_a : float
        time window of slow adaptation (seconds)
    alpha : float
        scaling factor between fast and slow adaptation
    baseline : float
        baseline neural response
    disable_long : bool
        disable the long term adaptation normalization pool (default: False)

    """

    def __init__(self, stim, sample_rate, shift, scale, tau, n, sigma, tau_a, tau_b, alpha, baseline, a=0, w=0, sigma_a=0, sf_bodies=None, sf_buildings=None,
                 sf_faces=None, sf_objects=None, sf_scenes=None, sf_scrambled=None, disable_long=False):

        # assign class variables
        self.shift = shift
        self.scale = scale
        self.tau = tau
        self.n = n
        self.w = w
        self.a = a
        self.sigma = sigma
        self.sigma_a = sigma_a
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.alpha = alpha
        self.baseline = baseline
        self.stimin = stim

        self.disable_long = disable_long

        self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
        # self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes]

        # image classes
        self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES', 'SCRAMBLED']
        # self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES']

        # iniate temporal variables
        self.numtimepts = len(stim)
        self.srate = sample_rate

        # compute timepoints
        self.t = np.arange(0, self.numtimepts) / self.srate

        # compute the impulse response function (used in the nominator, convolution of the stimulus) were m = 2
        #self.irf = self.gammaPDF(self.t, self.tau, 2) + self.a * (self.t + 1) ** -self.w
        #self.irf = self.gammaPDF(self.t, self.tau, 2) + self.gammaPDF(self.t, self.a, 2)
        #self.irf = self.gammaPDF(self.t, self.tau, 2)

        self.irf = gamma.pdf(self.t, self.tau, loc=1e-10)
        self.irf = self.irf / np.sum(self.irf) # Normalize the IRF

        #self.irf = self.diffgammaPDF(self.t, self.tau, self.w)
        # create long term adaptation exponential decay filter (for the normalization, convolution of the linear response)
        self.norm_irf = self.power_decay_fixed_window(self.t, self.tau_a, t_max=0.3)

        # create short term exponential decay filter
        #self.norm_irf_short = self.exponential_decay(self.t, self.tau_b)
        self.norm_irf_short = self.power_decay_fixed_window(self.t, self.tau_b, t_max=0.3)
    def scaling_stimulus(self, input, trial, cond, cat, root):
        """ Adapt stimulus height.

        params
        -----------------------
        input : float
            array containing values of input timecourse
        trial : string
            indicates type of trial (e.g. 'onepulse')
        cond : int
            ISI condition
        cat : str
            image category
        dir : str
            root directory

        returns
        -----------------------
        stim : float
            adapted response

        """

        # create copy of input
        stim = np.zeros(len(input))

        # determine which scaling factor to use
        cat_idx = self.stim.index(cat)
        sf = self.sf[cat_idx]

        # scale stimulus timecourse
        if 'onepulse' in trial:

            # import stimulus timepoints
            timepoints_onepulse = np.loadtxt(root + 'variables/timepoints_onepulse.txt', dtype=int)

            # define start and end of stimulus (expressed as timepts)
            start = timepoints_onepulse[cond, 0]
            end = timepoints_onepulse[cond, 1]

            # scale timecourse
            stim[start:end] = input[start:end] * sf

        elif 'twopulse' in trial:

            # import stimulus timepoints
            timepoints_twopulse = np.loadtxt(root + 'variables/timepoints_twopulse.txt', dtype=int)

            # define start and end of stimulus (expressed as timepts)
            start1 = timepoints_twopulse[cond, 0]
            end1 = timepoints_twopulse[cond, 1]
            start2 = timepoints_twopulse[cond, 2]
            end2 = timepoints_twopulse[cond, 3]

            # scale timecourse
            stim[start1:end1] = input[start1:end1] * sf
            stim[start2:end2] = input[start2:end2] * sf

        return stim

    def response_shift(self, input):
        """ Shifts response in time in the case that there is a delay betwween stimulus onset and response.

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        stim : float
            adapted response

        """

        # add shift to the stimulus
        # sft = self.shift/(1/self.srate)
        stimtmp = np.pad(input, (int(self.shift), 0), 'constant', constant_values=0)
        stim = stimtmp[0: self.numtimepts]

        return stim

    def lin(self, input):
        """ Convolves input with the Impulse Resone Function (irf)

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        linrsp : float
            adapted linear response

        """

        # compute the convolution
        linrsp = np.convolve(input, self.irf, 'full')
        linrsp = linrsp[0:self.numtimepts]

        return linrsp

    def rectf(self, input):
        """ Full-wave rectification of the input.

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        rectf : float
            adapted rectified response

        """

        rectf = abs(input)

        return rectf

    def exp(self, input):
        """ Exponentiation of the input.

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        exp : float
            adapted exponated response

        """

        exp = input ** self.n

        return exp

    # def norm(self, input, linrsp):
    #     """ Normalization of the input.

    #     params
    #     -----------------------
    #     input : float
    #         array containing values of input timecourse
    #     linrsp : float
    #         array containing values of linear response

    #     returns
    #     -----------------------
    #     rsp : float
    #         adapted response

    #     """

    #     # compute the normalized response
    #     demrsp = self.sigma**self.n + abs(linrsp)**self.n                       # semi-saturate + exponentiate
    #     normrsp = input/demrsp                                                  # divide

    #     # scale with gain
    #     rsp = self.scale * normrsp

    #     return rsp

    def vectorized_hold_signal(self,signal, mask):
        # Identify the start of each block of 1s
        starts = np.where((mask[:-1] == 0) & (mask[1:] == 1))[0] + 1
        # Identify the end of each block of 1s
        ends = np.where((mask[:-1] == 1) & (mask[1:] == 0))[0] + 1

        # Check if mask starts or ends with 1s and adjust starts/ends accordingly
        if mask[0] == 1:
            starts = np.insert(starts, 0, 0)
        if mask[-1] == 1:
            ends = np.append(ends, len(mask))

        # Create an array to hold the result
        held_signal = np.copy(signal)

        # Apply the hold operation vectorized
        for start, end in zip(starts, ends):
            held_signal[start:end] = signal[start]

        return held_signal

    def hold_signal_to_next_block(self, signal, mask):
        # Identify the start of each block of 1s
        starts = np.where((mask[:-1] == 0) & (mask[1:] == 1))[0] + 1

        # Check if mask starts with 1s and adjust starts accordingly
        if mask[0] == 1:
            starts = np.insert(starts, 0, 0)

        # Create an array to hold the result
        held_signal = np.copy(signal)

        # Apply the hold operation
        for i in range(len(starts) - 1):
            start = starts[i]
            next_start = starts[i + 1]
            held_signal[start:next_start] = signal[start]

        # Handle the last block
        if len(starts) > 0:
            held_signal[starts[-1]:] = signal[starts[-1]]

        return held_signal

    def norm_delay(self, input, linrsp):
        """ Introduces delay in input

        params
        -----------------------
        input : float
            array containing values of linear + rectf + exp
        linrsp : float
            array containing values of linear response

        returns
        -----------------------
        rsp : float
            adapted response

        """

        # compute the normalized delayed response

        poolrsp = np.convolve(linrsp, self.alpha*self.norm_irf, 'full')  # delay
        poolrsp = poolrsp[0:self.numtimepts]
        demrsp = self.sigma_a ** self.n + abs(poolrsp) ** self.n  # semi-saturate + exponentiate
        #demrsp = self.vectorized_hold_signal(demrsp, self.stimin)
        if self.disable_long:
            demrsp = self.hold_signal_to_next_block(demrsp, self.stimin)

        # TODO: Perhaps the short term adaptation should be exp/pow(+1) instead of exp(-1) --> Surpression increases over time
        poolrsp_short = np.convolve(linrsp, self.norm_irf_short, 'full')  # delay
        #poolrsp_short = self.apply_short_term_adaptation_irf(linrsp, self.norm_irf_short, self.stimin)
        poolrsp_short = poolrsp_short[0:self.numtimepts]
        # TODO: decouple sigma and n for short term adaptation from long term adaptation ?
        demrsp_short = self.sigma ** self.n + abs(poolrsp_short) ** self.n  # semi-saturate + exponentiate


        #combined_demrsp = demrsp + demrsp_short
        #normrsp = input / combined_demrsp  # divide

        normrsp = input / demrsp_short
        normrsp = normrsp / demrsp

        # scale with gain
        rsp = self.scale * normrsp + self.baseline

        return rsp, demrsp, demrsp_short

    def apply_short_term_adaptation_irf(self, signal, irf, stim):
        """
        Apply short-term adaptation impulse response function to stim blocks.

        Parameters:
        stim (np.array): Input array of 0s and 1s where 1s are stim blocks.
        irf (np.array): Impulse response function to be applied.

        Returns:
        np.array: Adapted response.
        """
        # Identify the start and end of each stim block
        stim_blocks = np.where(np.diff(stim, prepend=0, append=0) != 0)[0]
        stimstarts = stim_blocks[::2]
        stimends = np.concatenate((stimstarts[1:] - 1, [len(stim)]))

        # Initialize the response array
        response = np.zeros_like(stim, dtype=float)

        # Apply IRF to each stim block
        for i in range(len(stimstarts)):
            start = stimstarts[i]
            end = stimends[i]

            # Apply IRF to the current block
            block_response = np.convolve(signal[start:end], irf, mode='full')[:end - start]

            # Reset the adaptation response before the start of the next block
            response[start:end] = block_response

        return response

    def scaling_prediction(self, input, cat):
        """ Adapt stimulus height.

        params
        -----------------------
        input : float
            array containing values of input timecourse
        cat : str
            image category

        returns
        -----------------------
        stim : float
            adapted response

        """

        # determine which scaling factor to use
        cat_idx = self.stim.index(cat)
        sf = self.sf[cat_idx]

        # scale with gain
        rsp = sf * input

        return rsp

    def gammaPDF(self, t, tau, n):
        """ Returns values of a gamma function for a given timeseries.

        params
        -----------------------
        t : array dim(1, T)
            contains timepoints
        tau : float
            peak time
        n : int
            effects response decrease after peak

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains gamma values for each timepoint
        """

        y = (t / tau) ** (n - 1) * np.exp(-t / tau) / (tau * math.factorial(n - 1))
        y_norm = y / np.sum(y)

        return y_norm

    def diffgammaPDF(self, t, tau, w):
        """ Returns values of a gamma function for a given timeseries.

        params
        -----------------------
        t : array dim(1, T)
            contains timepoints
        tau : float
            peak time
        n : int
            effects response decrease after peak

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains gamma values for each timepoint
        """

        y = (t*np.exp(-t/tau)) - (w*t*np.exp(-t/(1.5*tau)))
        y_norm = y / np.sum(y)

        return y_norm

    def exponential_decay(self, t, tau):
        """ Impulse Response Function

        params
        -----------------------
        timepots : int
            length of timeseries
        tau : float
            peak time

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains value for each timepoint
        """

        y = np.exp(-t / tau)
        y_norm = y / np.sum(y)

        return y_norm

    def power_decay(self, t, tau):
        """ Impulse Response Function

        params
        -----------------------
        timepots : int
            length of timeseries
        tau : float
            peak time

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains value for each timepoint
        """
        epsilon = 1
        t = t + epsilon

        y = t**(-tau)
        y_norm = y / np.sum(y)

        return y_norm

    def power_decay_fixed_window(self, t, tau, t_max = 10.):
        """
        Fixed Window Power Decay Function

        params
        -----------------------
        t : array
            timepoints
        tau : float
            decay constant
        window_size : float
            size of the window for normalization (seconds)

        returns
        -----------------------
        y_norm : array
            normalized power decay values for each timepoint
        """
        if t_max > t[-1]:
            t_max = t[-1]
        num_points = int(t_max / t[1])

        epsilon = 1
        t = t + epsilon

        # Compute power decay
        y = t ** (-tau)

        # Normalize the decay function based on the fixed window size
        norm_factor = np.sum(y[:num_points])
        y_norm = y / norm_factor

        return y_norm

    def power_decay_truncated(self, t, tau, t_max=10.):
        """
        Truncated Impulse Response Function

        params
        -----------------------
        t : array
            timepoints
        tau : float
            decay constant
        t_max : float
            truncation time (seconds)

        returns
        -----------------------
        y_norm : array
            normalized truncated power decay values
        """

        # Check if t is long enough to reach t_max
        if t[-1] < t_max:
            t = np.append(t, np.arange(t[-1], t_max, t[1] - t[0]))

        epsilon = 1
        t = t + epsilon

        # Compute power decay
        y = t ** (-tau)

        # Truncate the time array to end at t_max
        truncated_indices = t <= t_max
        y_truncated = y[truncated_indices]

        # Normalize the truncated decay function
        y_norm = y_truncated / np.sum(y_truncated)

        return y_norm