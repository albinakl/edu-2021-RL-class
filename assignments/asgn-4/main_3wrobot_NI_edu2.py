#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator).

"""

import os, sys
PARENT_DIR = os.path.abspath(__file__ + '/../..')
sys.path.insert(0, PARENT_DIR)
import rcognita

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = f"this script is being run using " \
           f"rcognita ({rcognita.__version__}) " \
           f"located in cloned repository at '{PARENT_DIR}'. " \
           f"If you are willing to use your locally installed rcognita, " \
           f"run this script ('{os.path.basename(__file__)}') outside " \
           f"'rcognita/presets'."
else:
    info = f"this script is being run using " \
           f"locally installed rcognita ({rcognita.__version__}). " \
           f"Make sure the versions match."
print("INFO:", info)

import pathlib

import warnings
import csv
from datetime import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from rcognita import simulator
from rcognita import systems
from rcognita import controllers
from rcognita import loggers
from rcognita import visuals
from rcognita.utilities import on_key_press

from rcognita.utilities import rep_mat
from rcognita.utilities import uptria2vec
from rcognita.utilities import push_vec
from rcognita import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from numpy.linalg import lstsq
from numpy import reshape
import warnings
from grading_utilities import AnswerTracker

import argparse

#----------------------------------------EDUCATION

class CtrlOptPredRL:
    """
    Class of predictive optimal controllers, primarily model-predictive control and predictive reinforcement learning, that optimize a finite-horizon cost.

    Currently, the actor model is trivial: an action is generated directly without additional policy parameters.

    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled.
    mode : : string
        Controller mode. Currently available (:math:`\\rho` is the stage objective, :math:`\\gamma` is the discounting factor):

        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1

           * - Mode
             - Cost function
           * - 'MPC' - Model-predictive control (MPC)
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
           * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
             - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`
           * - 'SQL' - RL/ADP via stacked Q-learning
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\hat \\gamma^{k-1} Q^{\\theta}(y_{N_a}, u_{N_a})`

        Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.

        *Add your specification into the table when customizing the agent*.

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default).
    action_init : : array of shape ``[dim_input, ]``
        Initial action to initialize optimizers.
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).
    Nactor : : natural number
        Size of prediction horizon :math:`N_a`.
    pred_step_size : : number
        Prediction step size in :math:`J_a` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon.
    sys_rhs, sys_out : : functions
        Functions that represent the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
    prob_noise_pow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.
    is_est_model : : number
        Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`.
    model_est_stage : : number
        Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).
    model_est_period : : number
        Time between model estimate updates (in seconds).
    buffer_size : : natural number
        Size of the buffer to store data.
    model_order : : natural number
        Order of the state-space estimation model

        .. math::
            \\begin{array}{ll}
    			\\hat x^+ & = A \\hat x + B action, \\newline
    			observation^+  & = C \\hat x + D action.
            \\end{array}

        See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
        When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons``.
    model_est_checks : : natural number
        Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
        May improve the prediction quality somewhat.
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of stage objectives along horizon.
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
    critic_period : : number
        The same meaning as ``model_est_period``.
    critic_struct : : natural number
        Choice of the structure of the critic's features.

        Currently available:

        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1

           * - Mode
             - Structure
           * - 'quad-lin'
             - Quadratic-linear
           * - 'quadratic'
             - Quadratic
           * - 'quad-nomix'
             - Quadratic, no mixed terms
           * - 'quad-mix'
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`,
               where :math:`w` is the critic's weight vector

        *Add your specification into the table when customizing the critic*.
    stage_obj_struct : : string
        Choice of the stage objective structure.

        Currently available:

        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1

           * - Mode
             - Structure
           * - 'quadratic'
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
           * - 'biquadratic'
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
               should be ``[R1, R2]``

        *Pass correct stage objective parameters in* ``stage_obj_pars`` *(as a list)*

        *When customizing the stage objective, add your specification into the table above*

    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155

    """

    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 prob_noise_pow = 1,
                 is_est_model=0,
                 model_est_stage=1,
                 model_est_period=0.1,
                 buffer_size=20,
                 model_order=3,
                 model_est_checks=0,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 stage_obj_struct='quadratic',
                 stage_obj_pars=[],
                 observation_target=[]):
        """
        Parameters
        ----------
        dim_input, dim_output : : integer
            Dimension of input and output which should comply with the system-to-be-controlled.
        mode : : string
            Controller mode. Currently available (:math:`\\rho` is the stage objective, :math:`\\gamma` is the discounting factor):

            .. list-table:: Controller modes
               :widths: 75 25
               :header-rows: 1

               * - Mode
                 - Cost function
               * - 'MPC' - Model-predictive control (MPC)
                 - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
               * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
                 - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`
               * - 'SQL' - RL/ADP via stacked Q-learning
                 - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`

            Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.

            *Add your specification into the table when customizing the agent* .

        ctrl_bnds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default).
        action_init : : array of shape ``[dim_input, ]``
            Initial action to initialize optimizers.
        t0 : : number
            Initial value of the controller's internal clock
        sampling_time : : number
            Controller's sampling time (in seconds)
        Nactor : : natural number
            Size of prediction horizon :math:`N_a`
        pred_step_size : : number
            Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
            convenience. Larger prediction step size leads to longer factual horizon.
        sys_rhs, sys_out : : functions
            Functions that represent the right-hand side, resp., the output of the exogenously passed model.
            The latter could be, for instance, the true model of the system.
            In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
            Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
        prob_noise_pow : : number
            Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.
        is_est_model : : number
            Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`.
        model_est_stage : : number
            Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).
        model_est_period : : number
            Time between model estimate updates (in seconds).
        buffer_size : : natural number
            Size of the buffer to store data.
        model_order : : natural number
            Order of the state-space estimation model

            .. math::
                \\begin{array}{ll}
        			\\hat x^+ & = A \\hat x + B action, \\newline
        			observation^+  & = C \\hat x + D action.
                \\end{array}

            See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
            When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
            neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons``
        model_est_checks : : natural number
            Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
            May improve the prediction quality somewhat.
        gamma : : number in (0, 1]
            Discounting factor.
            Characterizes fading of stage objectives along horizon.
        Ncritic : : natural number
            Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
            optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
        critic_period : : number
            The same meaning as ``model_est_period``.
        critic_struct : : natural number
            Choice of the structure of the critic's features.

            Currently available:

            .. list-table:: Critic feature structures
               :widths: 10 90
               :header-rows: 1

               * - Mode
                 - Structure
               * - 'quad-lin'
                 - Quadratic-linear
               * - 'quadratic'
                 - Quadratic
               * - 'quad-nomix'
                 - Quadratic, no mixed terms
               * - 'quad-mix'
                 - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`,
                   where :math:`w` is the critic's weights

            *Add your specification into the table when customizing the critic*.
        stage_obj_struct : : string
            Choice of the stage objective structure.

            Currently available:

            .. list-table:: Running objective structures
               :widths: 10 90
               :header-rows: 1

               * - Mode
                 - Structure
               * - 'quadratic'
                 - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
               * - 'biquadratic'
                 - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
                   should be ``[R1, R2]``
        """

        self.dim_input = dim_input
        self.dim_output = dim_output

        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        # Controller: common
        self.Nactor = Nactor
        self.pred_step_size = pred_step_size

        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor)

        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)

        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )

        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys

        # Model estimator's things
        self.is_est_model = is_est_model
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow
        self.model_est_stage = model_est_stage
        self.model_est_period = model_est_period
        self.buffer_size = buffer_size
        self.model_order = model_order
        self.model_est_checks = model_est_checks

        A = np.zeros( [self.model_order, self.model_order] )
        B = np.zeros( [self.model_order, self.dim_input] )
        C = np.zeros( [self.dim_output, self.model_order] )
        D = np.zeros( [self.dim_output, self.dim_input] )
        x0est = np.zeros( self.model_order )

        self.my_model = models.ModelSS(A, B, C, D, x0est)

        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)

        # RL elements
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.stage_obj_struct = stage_obj_struct
        self.stage_obj_pars = stage_obj_pars
        self.observation_target = observation_target

        self.accum_obj_val = 0

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input) )
            self.Wmin = -1e3*np.ones(self.dim_critic)
            self.Wmax = 1e3*np.ones(self.dim_critic)
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 )
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3*np.ones(self.dim_critic)
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3*np.ones(self.dim_critic)
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_output + self.dim_output * self.dim_input + self.dim_input )
            self.Wmin = -1e3*np.ones(self.dim_critic)
            self.Wmax = 1e3*np.ones(self.dim_critic)

        self.w_critic_prev = np.ones(self.dim_critic)
        self.w_critic_init = self.w_critic_prev

        # self.big_number = 1e4

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.ctrl_clock = t0
        self.action_curr = self.action_min/10

    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state

    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.

        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])

        stage_obj = 0

        if self.stage_obj_struct == 'quadratic':
            R1 = self.stage_obj_pars[0]
            stage_obj = chi @ R1 @ chi
        elif self.stage_obj_struct == 'biquadratic':
            R1 = self.stage_obj_pars[0]
            R2 = self.stage_obj_pars[1]
            stage_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi

        return stage_obj

    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).

        """
        self.accum_obj_val += self.stage_obj(observation, action)*self.sampling_time

    def _estimate_model(self, t, observation):
        """
        Estimate model parameters by accumulating data buffers ``action_buffer`` and ``observation_buffer``.

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time: # New sample
            # Update buffers when using RL or requiring estimated model
            if self.is_est_model or self.mode in ['RQL', 'SQL']:
                time_in_est_period = t - self.est_clock

                # Estimate model if required
                if (time_in_est_period >= self.model_est_period) and self.is_est_model:
                    # Update model estimator's internal clock
                    self.est_clock = t

                    try:
                        # Using Github:CPCLAB-UNIPI/SIPPY
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                        SSest = sippy.system_identification(self.observation_buffer, self.action_buffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            # SS_f=int(self.buffer_size/12),
                                                            # SS_p=int(self.buffer_size/10),
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sampling_time)

                        self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)

                        # ToDo: train an NN via Torch
                        # NN_wgts = NN_train(...)

                    except:
                        print('Model estimation problem')
                        self.my_model.upd_pars(np.zeros( [self.model_order, self.model_order] ),
                                                np.zeros( [self.model_order, self.dim_input] ),
                                                np.zeros( [self.dim_output, self.model_order] ),
                                                np.zeros( [self.dim_output, self.dim_input] ) )

                    # Model checks
                    if self.model_est_checks > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        tot_abs_err_curr = 1e8
                        for k in range(self.model_est_checks):
                            A, B, C, D = self.model_stack[k].A, self.model_stack[k].B, self.model_stack[k].C, self.model_stack[k].D
                            x0est,_,_,_ = np.linalg.lstsq(C, observation)
                            Yest,_ = dss_sim(A, B, C, D, self.action_buffer, x0est, observation)
                            mean_err = np.mean(Yest - self.observation_buffer, axis=0)

                            # DEBUG ===================================================================
                            # ================================Interm output of model prediction quality
                            # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                            # dataRow = []
                            # for k in range(dim_output):
                            #     dataRow.append( mean_err[k] )
                            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                            # print( table )
                            # /DEBUG ===================================================================

                            tot_abs_err = np.sum( np.abs( mean_err ) )
                            if tot_abs_err <= tot_abs_err_curr:
                                tot_abs_err_curr = tot_abs_err
                                self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)

                        # DEBUG ===================================================================
                        # ==========================================Print quality of the best model
                        # R  = '\033[31m'
                        # Bl  = '\033[30m'
                        # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, observation)
                        # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.action_buffer, x0est, observation)
                        # mean_err = np.mean(Yest - ctrlStat.observation_buffer, axis=0)
                        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                        # dataRow = []
                        # for k in range(dim_output):
                        #     dataRow.append( mean_err[k] )
                        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                        # print(R+table+Bl)
                        # /DEBUG ===================================================================

            # Update initial state estimate
            x0est,_,_,_ = np.linalg.lstsq(self.my_model.C, observation)
            self.my_model.updateIC(x0est)

            if t >= self.model_est_stage:
                    # Drop probing noise
                    self.is_prob_noise = 0

    def _critic(self, observation, action, w_critic):
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.

        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])

        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi
        elif self.critic_struct == 'quad-mix':
            regressor_critic = np.concatenate([ observation**2, np.kron(observation, action), action**2 ])

        return w_critic @ regressor_critic

    def _critic_cost(self, w_critic):
        """
        Cost function of the critic.

        Currently uses value-iteration-like method.

        Customization
        -------------

        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation.

        """

        #YOUR CODE HERE


    def _critic_optimizer(self):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._critic_cost`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}

        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)

        w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_init, method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x

        # DEBUG ===================================================================
        # print('-----------------------Critic parameters--------------------------')
        # print( w_critic )
        # /DEBUG ==================================================================

        return w_critic

    def _actor_cost(self, action_sqn, observation):
        """
        See class documentation.

        Customization
        -------------

        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])

        observation_sqn = np.zeros([self.Nactor, self.dim_output])

        # System output prediction
        if not self.is_est_model:    # Via exogenously passed model
            observation_sqn[0, :] = observation
            state = self.state_sys
            for k in range(1, self.Nactor):
                # state = get_next_state(state, my_action_sqn[k-1, :], delta)         TODO
                state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k-1, :])  # Euler scheme

                observation_sqn[k, :] = self.sys_out(state)

        elif self.is_est_model:    # Via estimated model
            my_action_sqn_upsampled = my_action_sqn.repeat(int(self.pred_step_size/self.sampling_time), axis=0)
            observation_sqn_upsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, my_action_sqn_upsampled, self.my_model.x0est, observation)
            observation_sqn = observation_sqn_upsampled[::int(self.pred_step_size/self.sampling_time)]

        J = 0
        if self.mode=='MPC':
            for k in range(self.Nactor):
                J += self.gamma**k * self.stage_obj(observation_sqn[k, :], my_action_sqn[k, :])
        elif self.mode=='RQL':     # RL: Q-learning with Ncritic-1 roll-outs of stage objectives
             for k in range(self.Nactor-1):
                J += self.gamma**k * self.stage_obj(observation_sqn[k, :], my_action_sqn[k, :])
             J += self._critic(observation_sqn[-1, :], my_action_sqn[-1, :], self.w_critic)
        elif self.mode=='SQL':     # RL: stacked Q-learning
             for k in range(self.Nactor):
                Q = self._critic(observation_sqn[k, :], my_action_sqn[k, :], self.w_critic)

                # With state constraints via indicator function
                # Q = w_critic @ self._regressor_critic( observation_sqn[k, :], my_action_sqn[k, :] ) + state_constraint_indicator(observation_sqn[k, 0])

                # DEBUG ===================================================================
                # =========================================================================
                # R  = '\033[31m'
                # Bl  = '\033[30m'
                # if state_constraint_indicator(observation_sqn[k, 0]) > 1:
                #     print(R+str(state_constraint_indicator(observation_sqn[k, 0]))+Bl)
                # /DEBUG ==================================================================

                J += Q

        return J

    def _actor_optimizer(self, observation):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._actor_cost`.
        See class documentation.

        Customization
        -------------

        This method normally should not be altered, adjust :func:`~controllers.CtrlOptPred._actor_cost` instead.
        The only customization you might want here is regarding the optimization algorithm.

        """

        # For direct implementation of state constraints, this needs `partial` from `functools`
        # See [here](https://stackoverflow.com/questions/27659235/adding-multiple-constraints-to-scipy-minimize-autogenerate-constraint-dictionar)
        # def state_constraint(action_sqn, idx):

        #     my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])

        #     observation_sqn = np.zeros([idx, self.dim_output])

        #     # System output prediction
        #     if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
        #         observation_sqn[0, :] = observation
        #         state = self.state_sys
        #         Y[0, :] = observation
        #         x = self.x_s
        #         for k in range(1, idx):
        #             # state = get_next_state(state, my_action_sqn[k-1, :], delta)
        #             state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #             observation_sqn[k, :] = self.sys_out(state)

        #     return observation_sqn[-1, 1] - 1

        # my_constraints=[]
        # for my_idx in range(1, self.Nactor+1):
        #     my_constraints.append({'type': 'eq', 'fun': lambda action_sqn: state_constraint(action_sqn, idx=my_idx)})

        # my_constraints = {'type': 'ineq', 'fun': state_constraint}

        # Optimization method of actor
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        # actor_opt_method = 'SLSQP' # Standard
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}

        isGlobOpt = 0

        my_action_sqn_init = np.reshape(self.action_sqn_init, [self.Nactor*self.dim_input,])

        bnds = sp.optimize.Bounds(self.action_sqn_min, self.action_sqn_max, keep_feasible=True)

        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-7, 'options': actor_opt_options}
                action_sqn = basinhopping(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                          my_action_sqn_init,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter = 10).x
            else:
                action_sqn = minimize(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                      my_action_sqn_init,
                                      method=actor_opt_method,
                                      tol=1e-7,
                                      bounds=bnds,
                                      options=actor_opt_options).x

        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            action_sqn = my_action_sqn_init

        # DEBUG ===================================================================
        # ================================Interm output of model prediction quality
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])
        # my_action_sqn_upsampled = my_action_sqn.repeat(int(delta/self.sampling_time), axis=0)
        # observation_sqn_upsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, my_action_sqn_upsampled, self.my_model.x0est, observation)
        # observation_sqn = observation_sqn_upsampled[::int(delta/self.sampling_time)]
        # Yt = np.zeros([N, self.dim_output])
        # Yt[0, :] = observation
        # state = self.state_sys
        # for k in range(1, Nactor):
        #     state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #     Yt[k, :] = self.sys_out(state)
        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
        # dataRow = []
        # for k in range(dim_output):
        #     dataRow.append( np.mean(observation_sqn[:,k] - Yt[:,k]) )
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
        # print(R+table+Bl)
        # /DEBUG ==================================================================

        return action_sqn[:self.dim_input]    # Return first action

    def compute_action(self, t, observation):
        """
        Main method. See class documentation.

        Customization
        -------------

        Add your modes, that you introduced in :func:`~controllers.CtrlOptPred._actor_cost`, here.

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t

            if self.mode == 'MPC':

                # Apply control when model estimation phase is over
                if self.is_prob_noise and self.is_est_model:
                    return self.prob_noise_pow * (rand(self.dim_input) - 0.5)

                elif not self.is_prob_noise and self.is_est_model:
                    action = self._actor_optimizer(observation)

                elif self.mode=='MPC':
                    action = self._actor_optimizer(observation)

            elif self.mode in ['RQL', 'SQL']:
                # Critic
                timeInCriticPeriod = t - self.critic_clock

                # Update data buffers
                self.action_buffer = push_vec(self.action_buffer, self.action_curr)
                self.observation_buffer = push_vec(self.observation_buffer, observation)

                if timeInCriticPeriod >= self.critic_period:
                    # Update critic's internal clock
                    self.critic_clock = t

                    self.w_critic = self._critic_optimizer()
                    self.w_critic_prev = self.w_critic

                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.w_critic_init = self.w_critic_prev

                else:
                    self.w_critic = self.w_critic_prev

                # Actor. Apply control when model estimation phase is over
                if self.is_prob_noise and self.is_est_model:
                    action = self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                elif not self.is_prob_noise and self.is_est_model:
                    action = self._actor_optimizer(observation)

                elif self.mode in ['RQL', 'SQL']:
                    action = self._actor_optimizer(observation)

            self.action_curr = action

            return action

        else:
            return self.action_curr

#----------------------------------------/EDUCATION

#----------------------------------------Set up dimensions
dim_state = 3
dim_input = 2
dim_output = dim_state
dim_disturb = 2

dim_R1 = dim_output + dim_input
dim_R2 = dim_R1

description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['manual',
                             'nominal',
                             'MPC',
                             'RQL',
                             'SQL',
                             'JACS'],
                    default='MPC',
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----nominal: nominal controller, usually used to benchmark optimal controllers;' +
                    '----MPC:model-predictive control; ' +
                    '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; ' +
                    '----SQL: stacked Q-learning; ' +
                    '----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.')
parser.add_argument('--dt', type=float, metavar='dt',
                    default=0.01,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=3.0,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=['5', '5', '-3*pi/4'],
                    help='Initial state (as sequence of numbers); ' +
                    'dimension is environment-specific!')
parser.add_argument('--is_log_data', type=bool,
                    default=True,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=bool,
                    default=True,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=bool,
                    default=True,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--is_est_model', type=bool,
                    default=False,
                    help='Flag to estimate environment model.')
parser.add_argument('--model_est_stage', type=float,
                    default=1.0,
                    help='Seconds to learn model until benchmarking controller kicks in.')
parser.add_argument('--model_est_period_multiplier', type=float,
                    default=1,
                    help='Model is updated every model_est_period_multiplier times dt seconds.')
parser.add_argument('--model_order', type=int,
                    default=5,
                    help='Order of state-space estimation model.')
parser.add_argument('--prob_noise_pow', type=float,
                    default=False,
                    help='Power of probing (exploration) noise.')
parser.add_argument('--action_manual', type=float,
                    default=[-5, -3], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=3,
                    help='Horizon length (in steps) for predictive controllers.')
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=1.0,
                    help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
parser.add_argument('--buffer_size', type=int,
                    default=10,
                    help='Size of the buffer (experience replay) for model estimation, agent learning etc.')
parser.add_argument('--stage_obj_struct', type=str,
                    default='quadratic',
                    choices=['quadratic',
                             'biquadratic'],
                    help='Structure of stage objective function.')
parser.add_argument('--R1_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of stage objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of stage objective function . Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                    'where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--Ncritic', type=int,
                    default=4,
                    help='Critic stack size (number of temporal difference terms in critic cost).')
parser.add_argument('--gamma', type=float,
                    default=1.0,
                    help='Discount factor.')
parser.add_argument('--critic_period_multiplier', type=float,
                    default=1.0,
                    help='Critic is updated every critic_period_multiplier times dt seconds.')
parser.add_argument('--critic_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix'],
                    help='Feature structure (critic). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms; ' +
                    '----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations).')
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix'],
                    help='Feature structure (actor). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms.')

args = parser.parse_args()

#----------------------------------------Post-processing of arguments
# Convert `pi` to a number pi
for k in range(len(args.state_init)):
    args.state_init[k] = eval( args.state_init[k].replace('pi', str(np.pi)) )

args.state_init = np.array(args.state_init)
args.action_manual = np.array(args.action_manual)

pred_step_size = args.dt * args.pred_step_size_multiplier
model_est_period = args.dt * args.model_est_period_multiplier
critic_period = args.dt * args.critic_period_multiplier

R1 = np.diag(np.array(args.R1_diag))
R2 = np.diag(np.array(args.R2_diag))

assert args.t1 > args.dt > 0.0
assert args.state_init.size == dim_state

globals().update(vars(args))

#----------------------------------------(So far) fixed settings
is_disturb = 1
is_dyn_ctrl = 0

t0 = 0

action_init = 0 * np.ones(dim_input)

# Solver
atol = 1e-5
rtol = 1e-3

# xy-plane
xMin = -10
xMax = 10
yMin = -10
yMax = 10

# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0

# Control constraints
v_min = -25
v_max = 25
omega_min = -5
omega_max = 5
ctrl_bnds=np.array([[v_min, v_max], [omega_min, omega_max]])

#----------------------------------------Initialization : : system
my_sys = systems.Sys3WRobotNI(sys_type="diff_eqn",
                                     dim_state=dim_state,
                                     dim_input=dim_input,
                                     dim_output=dim_output,
                                     dim_disturb=dim_disturb,
                                     pars=[],
                                     ctrl_bnds=ctrl_bnds,
                                     is_dyn_ctrl=is_dyn_ctrl,
                                     is_disturb=is_disturb,
                                     pars_disturb = np.array([[20*dt, 20*dt], [0, 0], [0.3, 0.3]]))

observation_init = my_sys.out(state_init)

xCoord0 = state_init[0]
yCoord0 = state_init[1]
alpha0 = state_init[2]
alpha_deg_0 = alpha0/2/np.pi

#----------------------------------------Initialization : : model

#----------------------------------------Initialization : : controller
my_ctrl_nominal = controllers.CtrlNominal3WRobotNI(ctrl_gain=0.5, ctrl_bnds=ctrl_bnds, t0=t0, sampling_time=dt)

# Predictive optimal controller
#controllers.
my_ctrl_opt_pred = CtrlOptPredRL(dim_input,
                                           dim_output,
                                           ctrl_mode,
                                           ctrl_bnds = ctrl_bnds,
                                           action_init = [],
                                           t0 = t0,
                                           sampling_time = dt,
                                           Nactor = Nactor,
                                           pred_step_size = pred_step_size,
                                           sys_rhs = my_sys._state_dyn,
                                           sys_out = my_sys.out,
                                           state_sys = state_init,
                                           prob_noise_pow = prob_noise_pow,
                                           is_est_model = is_est_model,
                                           model_est_stage = model_est_stage,
                                           model_est_period = model_est_period,
                                           buffer_size = buffer_size,
                                           model_order = model_order,
                                           model_est_checks = model_est_checks,
                                           gamma = gamma,
                                           Ncritic = Ncritic,
                                           critic_period = critic_period,
                                           critic_struct = critic_struct,
                                           stage_obj_struct = stage_obj_struct,
                                           stage_obj_pars = [R1],
                                           observation_target = [])

# Stabilizing RL agent
my_ctrl_RL_stab = controllers.CtrlRLStab(dim_input,
                                         dim_output,
                                         ctrl_mode,
                                         ctrl_bnds = ctrl_bnds,
                                         action_init = action_init,
                                         t0 = t0,
                                         sampling_time = dt,
                                         Nactor = Nactor,
                                         pred_step_size = pred_step_size,
                                         sys_rhs = my_sys._state_dyn,
                                         sys_out = my_sys.out,
                                         state_sys = state_init,
                                         prob_noise_pow = prob_noise_pow,
                                         is_est_model = is_est_model,
                                         model_est_stage = model_est_stage,
                                         model_est_period = model_est_period,
                                         buffer_size = buffer_size,
                                         model_order = model_order,
                                         model_est_checks = model_est_checks,
                                         gamma = gamma,
                                         Ncritic = Ncritic,
                                         critic_period = critic_period,
                                         critic_struct = critic_struct,
                                         actor_struct = actor_struct,
                                         stage_obj_struct = stage_obj_struct,
                                         stage_obj_pars = [R1],
                                         observation_target = [],
                                         safe_ctrl = my_ctrl_nominal,
                                         safe_decay_rate = 1e-4)

if ctrl_mode == 'JACS':
    my_ctrl_benchm = my_ctrl_RL_stab
else:
    my_ctrl_benchm = my_ctrl_opt_pred

#----------------------------------------Initialization : : simulator
my_simulator = simulator.Simulator(sys_type = "diff_eqn",
                                   closed_loop_rhs = my_sys.closed_loop_rhs,
                                   sys_out = my_sys.out,
                                   state_init = state_init,
                                   disturb_init = np.array([0, 0]),
                                   action_init = action_init,
                                   t0 = t0,
                                   t1 = t1,
                                   dt = dt,
                                   max_step = dt/2,
                                   first_step = 1e-6,
                                   atol = atol,
                                   rtol = rtol,
                                   is_disturb = is_disturb,
                                   is_dyn_ctrl = is_dyn_ctrl)

#----------------------------------------Initialization : : logger
if os.path.basename( os.path.normpath( os.path.abspath(os.getcwd()) ) ) == 'presets':
    data_folder = '../simdata'
else:
    data_folder = 'simdata'

pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh%Mm%Ss")
datafiles = [None] * Nruns

for k in range(0, Nruns):
    datafiles[k] = data_folder + '/' + my_sys.name + '__' + ctrl_mode + '__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)

    if is_log_data:
        print('Logging data to:    ' + datafiles[k])

        with open(datafiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['System', my_sys.name ] )
            writer.writerow(['Controller', ctrl_mode ] )
            writer.writerow(['dt', str(dt) ] )
            writer.writerow(['state_init', str(state_init) ] )
            writer.writerow(['is_est_model', str(is_est_model) ] )
            writer.writerow(['model_est_stage', str(model_est_stage) ] )
            writer.writerow(['model_est_period_multiplier', str(model_est_period_multiplier) ] )
            writer.writerow(['model_order', str(model_order) ] )
            writer.writerow(['prob_noise_pow', str(prob_noise_pow) ] )
            writer.writerow(['Nactor', str(Nactor) ] )
            writer.writerow(['pred_step_size_multiplier', str(pred_step_size_multiplier) ] )
            writer.writerow(['buffer_size', str(buffer_size) ] )
            writer.writerow(['stage_obj_struct', str(stage_obj_struct) ] )
            writer.writerow(['R1_diag', str(R1_diag) ] )
            writer.writerow(['R2_diag', str(R2_diag) ] )
            writer.writerow(['Ncritic', str(Ncritic) ] )
            writer.writerow(['gamma', str(gamma) ] )
            writer.writerow(['critic_period_multiplier', str(critic_period_multiplier) ] )
            writer.writerow(['critic_struct', str(critic_struct) ] )
            writer.writerow(['actor_struct', str(actor_struct) ] )
            writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'stage_obj', 'accum_obj', 'v [m/s]', 'omega [rad/s]'] )

# Do not display annoying warnings when print is on
if is_print_sim_step:
    warnings.filterwarnings('ignore')

my_logger = loggers.Logger3WRobotNI()

#----------------------------------------Main loop
if is_visualization:

    state_full_init = my_simulator.state_full

    my_animator = visuals.Animator3WRobotNI(objects=(my_simulator,
                                                     my_sys,
                                                     my_ctrl_nominal,
                                                     my_ctrl_benchm,
                                                     datafiles,
                                                     controllers.ctrl_selector,
                                                     my_logger),
                                            pars=(state_init,
                                                  action_init,
                                                  t0,
                                                  t1,
                                                  state_full_init,
                                                  xMin,
                                                  xMax,
                                                  yMin,
                                                  yMax,
                                                  ctrl_mode,
                                                  action_manual,
                                                  v_min,
                                                  omega_min,
                                                  v_max,
                                                  omega_max,
                                                  Nruns,
                                                    is_print_sim_step, is_log_data, 0, []))

    anm = animation.FuncAnimation(my_animator.fig_sim,
                                  my_animator.animate,
                                  init_func=my_animator.init_anim,
                                  blit=False, interval=dt/1e6, repeat=False)

    my_animator.get_anm(anm)

    cId = my_animator.fig_sim.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, anm))

    anm.running = True

    my_animator.fig_sim.tight_layout()

    plt.show()

else:
    run_curr = 1
    datafile = datafiles[0]

    while True:

        my_simulator.sim_step()

        t, state, observation, state_full = my_simulator.get_sim_step_data()

        action = controllers.ctrl_selector(t, observation, action_manual, my_ctrl_nominal, my_ctrl_benchm, ctrl_mode)

        my_sys.receive_action(action)
        my_ctrl_benchm.receive_sys_state(my_sys._state)
        my_ctrl_benchm.upd_accum_obj(observation, action)

        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]

        stage_obj = my_ctrl_benchm.stage_obj(observation, action)
        accum_obj = my_ctrl_benchm.accum_obj_val

        if is_print_sim_step:
            my_logger.print_sim_step(t, xCoord, yCoord, alpha, stage_obj, accum_obj, action)

        if is_log_data:
            my_logger.log_data_row(datafile, t, xCoord, yCoord, alpha, stage_obj, accum_obj, action)

        if t >= t1:
            if is_print_sim_step:
                print('.....................................Run {run:2d} done.....................................'.format(run = run_curr))

            run_curr += 1

            if run_curr > Nruns:
                break

            if is_log_data:
                datafile = datafiles[run_curr-1]

            # Reset simulator
            my_simulator.status = 'running'
            my_simulator.t = t0
            my_simulator.observation = state_full_init

            if ctrl_mode != 'nominal':
                my_ctrl_benchm.reset(t0)
            else:
                my_ctrl_nominal.reset(t0)

            accum_obj = 0
