"""
BEng Final Year Project: Machine Vision Calibration System
==========================================================

Python port of the MATLAB algorithms from J. Minton's 2003 BEng thesis
on 2D digital image correlation and machine vision plane calibration.

The system operates in two stages:
  Stage 1 -- Position the camera relative to the calibration plane
             (marker detection and angle estimation)
  Stage 2 -- Find object displacement using DIC with rigid-body compensation

Modules
-------
dic          : 2D cross-correlation via FFT
markers      : Colour-based marker detection pipeline
calibration  : Angle calibration and polynomial fitting
pipeline     : Full end-to-end pipeline combining all stages
"""
