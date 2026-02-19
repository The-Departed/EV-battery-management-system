"""
Battery Thermal Modeling - Data Generation Package
===================================================

This package contains the complete synthetic data generation pipeline
for battery thermal modeling and machine learning.

Modules
-------
- drive_cycles: Standard drive cycle current profiles (UDDS, US06, HWFET)
- temperature_profiles: Ambient temperature scenario generator
- batch_simulator: Physics-based batch simulation (ECM + EETM)
- sensor_noise: Realistic BMS sensor noise injection
- dataset_builder: Large-scale ML dataset generation
- validation: Dataset quality and sanity checks

Usage
-----
Each module can be run independently:
    python -m generation.drive_cycles
    python -m generation.temperature_profiles
    python -m generation.batch_simulator
    python -m generation.sensor_noise
    python -m generation.dataset_builder
    python -m generation.validation

Or run the complete pipeline:
    python -m generation.pipeline

Author: Battery Modeling Pipeline
Date: 2026-01-27
Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Battery Modeling Pipeline"
__all__ = [
    "DriveCycleLoader",
    "TemperatureProfileGenerator", 
    "BatchPhysicsSimulator",
    "SensorNoiseInjector",
    "DatasetBuilder",
    "DatasetValidator"
]
