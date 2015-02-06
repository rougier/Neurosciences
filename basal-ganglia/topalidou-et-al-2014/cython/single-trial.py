# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
from model import *
from display import *

# Make GPI lesion
# connections["GPI.cog -> THL.cog"].active = False
# connections["GPI.mot -> THL.mot"].active = False

reset()
for i in xrange(  0, 500): iterate(dt)
set_trial()
for i in xrange(500,3000): iterate(dt)


# -----------------------------------------------------------------------------
history = np.zeros(3000, dtype=htype)
history["CTX"]["mot"] = CTX.mot.history[:3000]
history["CTX"]["cog"] = CTX.cog.history[:3000]
history["CTX"]["ass"] = CTX.ass.history[:3000]
history["STR"]["mot"] = STR.mot.history[:3000]
history["STR"]["cog"] = STR.cog.history[:3000]
history["STR"]["ass"] = STR.ass.history[:3000]
history["STN"]["mot"] = STN.mot.history[:3000]
history["STN"]["cog"] = STN.cog.history[:3000]
history["GPI"]["mot"] = GPI.mot.history[:3000]
history["GPI"]["cog"] = GPI.cog.history[:3000]
history["THL"]["mot"] = THL.mot.history[:3000]
history["THL"]["cog"] = THL.cog.history[:3000]

if 1: display_ctx(history, 3.0, "single-trial.pdf")
if 0: display_all(history, 3.0, "single-trial-all.pdf")
