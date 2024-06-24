## Monitoring Data Drift with Evidently AI and Grafana

Welcome to the repository for monitoring data drift using Evidently AI and Grafana. This guide will help you set up a robust system to ensure your machine learning models remain accurate and reliable over time.

### Overview

Data drift can significantly impact the performance of machine learning models. This project provides a comprehensive solution to monitor data drift by leveraging Evidently AI for generating metrics and Grafana for visualization. The setup uses Docker Compose for easy deployment and management of services.

### Features
* Data Drift Monitoring: Keep track of changes in data distribution and ensure your models remain performant.
* Visualization: Use Grafana to visualize drift metrics and gain insights into data quality.
* Easy Setup: Deploy PostgreSQL, Adminer, and Grafana using Docker Compose.

### Setup 

Create virtual environment:

```
python -m venv venv
```

Activate the virtual environment:
* On Windows:
```
myenv\Scripts\activate
```
* On macOS and Linux:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Install new package if needed:

```
pip install {package_name}
pip freeze > requirements.txt
```

Deactivate the virtual environment:
```
deactivate
```

Don't forget to activate the virtual environment every time you enter the project again.

### Run 

```
docker-compose up --build
```

Grafana is running on `http://localhost:3000`

Database is running on `http://localhost:8080/`

Run the data generation script:

```
python generate_data.py
```

To stop:

```
docker-compose down
```

To access Evidently for monitoring:

```
evidently ui  
```

![Here is the example visualization](./Grafana_monitoring.png)