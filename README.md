# Multi-Agent Urban Traffic Signal Optimization in Simulated Environments

**Project ID:** 25-26J-523  
**Specialization:** Software Engineering (SE)  
**Research Group:** CoEAI - Centre of Excellence for AI

---

## 📖 About the Project

Urban traffic congestion is a critical issue affecting commute times, fuel consumption, and air quality. Traditional fixed-time traffic signals often fail to adapt to real-time conditions or handle non-recurring events like accidents. Furthermore, current emergency vehicle preemption systems often lack network-wide coordination.

This project proposes a **modular, city-scale traffic signal optimization framework** that integrates cutting-edge AI and control methodologies. By combining **Graph Neural Networks (GNN)** and **Model Predictive Control (MPC)** within a realistic **SUMO simulation**, we aim to create a system that dynamically adapts to traffic flows and ensures the rapid, safe passage of emergency vehicles.

## 🚀 Key Features

* **Realistic Simulation:** A configurable city traffic model built on SUMO with a custom web-based control panel.
* **AI-Driven Optimization:** Traffic signal timing optimization using Graph Neural Networks (GNN) based on spatial and temporal traffic patterns.
* **Predictive Control:** An alternative Model Predictive Control (MPC) strategy for computing optimal signal phases over a prediction horizon.
* **Emergency Preemption:** A dynamic "Green Wave" system for emergency vehicles, featuring route prediction and conflict minimization.
* **Interactive Dashboards:** Real-time visualization and control interfaces for all modules.

-----

## 🛠️ Technology Stack

  * **Simulation:** SUMO, TraCI
  * **Machine Learning:** Python, PyTorch Geometric (GNN), Reinforcement Learning
  * **Control Theory:** Model Predictive Control (MPC)
  * **Frontend/Dashboards:** React.js, Interactive Data Visualization Libraries
  * **Backend:** Node.js, Python (Flask/FastAPI)
  * **Mobile:** React Native / Flutter (for EV Driver & Public User apps)

-----

## 🎯 Objectives

1.  **Develop** a realistic city-scale traffic simulation environment using SUMO.
2.  **Optimize** traffic signal timings using a GNN-based approach that considers spatial and temporal relationships.
3.  **Implement** an MPC-based controller to provide robust, constraint-aware signal management.
4.  **Ensure** rapid emergency response through a smart preemption system that dynamically clears paths for emergency vehicles.
5.  **Visualize** system performance through comprehensive, real-time dashboards.

-----

*This project is conducted as part of the final year research requirement at SLIIT.*

```
```