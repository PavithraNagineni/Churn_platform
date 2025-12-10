# 🔥 Churn Prediction Platform (Deep Learning + FastAPI)

A production-ready customer churn prediction system built using **PyTorch**, **FastAPI**, and **Docker**.  
This project includes preprocessing, deep learning model training, evaluation, and a real-time inference API.

---

## 🚀 Features
- Deep Learning model (PyTorch) for churn prediction  
- Modular preprocessing pipeline  
- FastAPI inference microservice  
- Dockerized deployment  
- Clean, industry-ready folder structure  
- Supports real-time predictions  

---

## 🏗 Project Architecture

           +----------------------+
           |     Raw Dataset      |
           +----------+-----------+
                      |
                      v
    +---------------------------------------+
    |      Preprocessing Pipeline           |
    |  (Scaling, Cleaning, Feature Prep)    |
    +------------------+--------------------+
                       |
                       v
     +-------------------------------------+
     |      PyTorch ChurnNet Model         |
     | (Training, Validation, Evaluation)  |
     +------------------+------------------+
                       |
                       v
            +----------------------+
            |   Saved Model (.pt) |
            +----------+-----------+
                       |
                       v
    +--------------------------------------+
    |      FastAPI Inference Microservice  |
    |  (/predict endpoint returns churn)   |
    +------------------+-------------------+
                       |
                       v
            +----------------------+

            |   Docker Deployment |
            +----------------------+

            

## 📁 Folder Structure
churn-platform/

│── data/

│── models/

│── src/

│ ├── config.py

│ ├── data_preprocessing.py

│ ├── model.py

│ ├── train.py

│ └── evaluate.py

│── service/

│ ├── app.py

│ └── schemas.py

│── requirements.txt

│── Dockerfile

│── README.md

<img width="920" height="650" alt="image" src="https://github.com/user-attachments/assets/d1c70956-a62d-466c-b14b-d4b2489f310c" />


<img width="1337" height="637" alt="image" src="https://github.com/user-attachments/assets/c86bf3be-5163-4c70-a827-303532f1922c" />


<img width="1202" height="623" alt="image" src="https://github.com/user-attachments/assets/8d8aec71-8065-4dcb-9712-d1a8b5c08a47" />


