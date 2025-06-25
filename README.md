# 🤖 Robot Arm Learning Task – FEUP

This project involves training a robotic arm using **Reinforcement Learning** (Proximal Policy Optimization – PPO) with the `robosuite` simulator to grasp a cube and move it to another location (another table surface). Developed as part of a robotics and AI course at **FEUP**.

---

## 📁 Project Structure
```
Root/
├── DLR/
│   └── network.py              # PPO agent and network
├── env/
│   └── PickMove.py             # Custom environment
├── models/                     # Saved models (checkpoints, final_model.pth)
├── run.py                      # Training execution script
├── check.py                    # Model evaluation / debugging
├── requirements.txt            # Dependencies
├── README.txt                  # This file
└── .gitignore                  # Git ignore list
```

## 📦 Dependencies

To install all required packages:
```
pip install -r requirements.txt
```
