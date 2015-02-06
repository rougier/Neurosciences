# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------

# --- Time ---
ms           = 0.001
dt           = 1*ms
tau          = 10*ms

# --- Learning ---
alpha_CUE  = 0.05
alpha_LTP  = 0.002
alpha_LTD  = 0.001

# --- Sigmoid ---
Vmin = 0
Vmax = 20
Vh   = 16
Vc   = 3

# --- Model ---
decison_threshold  = 40
noise              = 0.001
CTX_rest   =  -3.0
STR_rest   =   0.0
STN_rest   = -10.0
GPI_rest   =  10.0
THL_rest   = -40.0

# --- Cues & Rewards ---
V_cue   = 7
rewards = 3/3.,2/3.,1/3.,0/3.

# -- Weight ---
Wmin  = 0.25
Wmax  = 0.75
gains = { "CTX.cog -> STR.cog" : +1.0,
          "CTX.mot -> STR.mot" : +1.0,
          "CTX.ass -> STR.ass" : +1.0,
          "CTX.cog -> STR.ass" : +0.2,
          "CTX.mot -> STR.ass" : +0.2,
          "CTX.cog -> STN.cog" : +1.0,
          "CTX.mot -> STN.mot" : +1.0,
          "STR.cog -> GPI.cog" : -2.0,
          "STR.mot -> GPI.mot" : -2.0,
          "STR.ass -> GPI.cog" : -2.0,
          "STR.ass -> GPI.mot" : -2.0,
          "STN.cog -> GPI.cog" : +1.0,
          "STN.mot -> GPI.mot" : +1.0,
          "GPI.cog -> THL.cog" : -0.25,
          "GPI.mot -> THL.mot" : -0.25,

          "THL.cog -> CTX.cog" : +0.4,
          "THL.mot -> CTX.mot" : +0.4,
          "CTX.cog -> THL.cog" : +0.1,
          "CTX.mot -> THL.mot" : +0.1,

          "CTX.mot -> CTX.mot" : +0.5,
          "CTX.cog -> CTX.cog" : +0.5,
          "CTX.ass -> CTX.ass" : +0.5,

          "CTX.ass -> CTX.cog" : +0.01,
          "CTX.ass -> CTX.mot" : +0.01,
          "CTX.cog -> CTX.ass" : +0.01,
          "CTX.mot -> CTX.ass" : +0.01,
 }
