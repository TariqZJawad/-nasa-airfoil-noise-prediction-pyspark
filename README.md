<div align="center">
# ✈️ AeroAcoustic ML: Predicting Airfoil Self-Noise with Apache Spark
**An End-to-End Big Data ETL & Machine Learning Pipeline using PySpark**
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark%203.1-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Apache Parquet](https://img.shields.io/badge/Storage-Apache%20Parquet-40B5A4?style=for-the-badge&logo=apache&logoColor=white)](https://parquet.apache.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)
</div>
---
## 📖 1. The Engineering Challenge (The Story)
In modern aeronautics and high-performance automotive engineering, **aerodynamic noise** is a critical design constraint. As air flows over an airfoil (such as an aircraft wing, turbine blade, or sports car spoiler), turbulence interacts with the blade's trailing edge, generating **airfoil self-noise**.
Testing every new wing prototype inside an acoustic wind tunnel is **expensive, time-consuming, and resource-intensive**. 
**💡 The Solution:**  
Instead of relying solely on physical wind tunnel tests for every design iteration, this project builds a **scalable Machine Learning Pipeline using Apache Spark (PySpark)**. By leveraging historical wind tunnel data from **NASA**, our pipeline processes raw aerodynamic measurements, standardizes the features, and trains a predictive model capable of estimating the **Sound Pressure Level (in Decibels)** of an airfoil before it is physically manufactured.
<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Airfoil_angle_of_attack.svg/800px-Airfoil_angle_of_attack.svg.png" width="48%" alt="Airfoil Angle of Attack Diagram">
  <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/images/Airfoil_with_flow.png" width="45%" alt="Airfoil Airflow Diagram">
  <p><em>Physical geometry of an airfoil showing airflow behavior and the Angle of Attack (α).</em></p>
</div>
---
## 🏗️ 2. Pipeline Architecture & Workflow
To ensure the solution can scale from thousands to millions of sensor readings in a distributed environment, the entire workflow is engineered using **PySpark SQL** and **PySpark MLlib**:
<div align="center">

| **1. Raw Data** | ➡️ | **2. Spark ETL** | ➡️ | **3. Feature Engineering** | ➡️ | **4. ML Training** | ➡️ | **5. Deployment** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CSV` (1,522 rows) |  | Deduplicate & Drop Nulls → `Parquet` (1,499 rows) |  | `VectorAssembler` + `StandardScaler` |  | `LinearRegression` Model |  | Persisted `PipelineModel` |

</div>
---
## 🛠️️ 3. Data Engineering & ETL Journey
The dataset used in this project is derived from the **NASA Airfoil Self-Noise Dataset** (obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel).
### 🔹 Feature Dictionary

| Column Name | Unit | Aerodynamic Description | Role |
| :--- | :--- | :--- | :--- |
| **`Frequency`** | Hz | Acoustic frequency of the generated sound wave. | Input Feature |
| **`AngleOfAttack`** | Degrees (°) | Angle between the airfoil chord line and oncoming airflow ($\alpha$). | Input Feature |
| **`ChordLength`** | Meters (m) | Distance between the leading edge and trailing edge of the airfoil. | Input Feature |
| **`FreeStreamVelocity`** | m/s | Velocity of the airflow entering the wind tunnel. | Input Feature |
| **`SuctionSideDisplacement`** | Meters (m) | Boundary layer displacement thickness on the suction side. | Input Feature |
| **`SoundLevelDecibels`** | dB | Scaled Sound Pressure Level (Target to predict). | **Target Label** |

### 🔹 Raw Data Inspection
Below is a sample of the ingested raw telemetry data before transformation:
![Raw Data Sample](raw_data_sample.png)
### 🔹 Data Quality & Optimization Metrics
During the **ETL (Extract, Transform, Load)** phase, the dataset underwent strict quality checks before being converted to **Apache Parquet** for optimized columnar storage and faster I/O performance:
* **Initial Raw Records:** `1,522` rows
* **After Deduplication (`dropDuplicates`):** `1,503` rows *(19 duplicate records removed)*
* **After Null Removal (`dropna`):** `1,499` clean rows *(4 incomplete records removed)*
* **Schema Standardization:** Renamed target column from `SoundLevel` to `SoundLevelDecibels`.
---
## ⚙️ 4. Machine Learning Pipeline Construction
To prevent data leakage and ensure seamless deployment, feature transformations and model training were encapsulated into a unified **3-Stage Spark ML Pipeline** (trained on **70%** of the data and tested on **30%** with `seed=42`):
1. **Stage 1 — `VectorAssembler`:** Consolidates the 5 physical and aerodynamic input columns into a single dense feature vector (`features`).
2. **Stage 2 — `StandardScaler`:** Because features vary drastically in scale (e.g., `Frequency` in thousands of Hz vs. `SuctionSideDisplacement` in thousandths of a meter), this stage normalizes all features to unit standard deviation (`scaledFeatures`).
3. **Stage 3 — `LinearRegression`:** Fits a multivariate linear regression model mapping the scaled features to `SoundLevelDecibels`.
---
## 📊 5. Model Evaluation & Physical Insights
### 🔹 Regression Performance on Unseen Test Data

| Evaluation Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **`3.73 dB`** | On average, the model's predictions deviate by only ~3.73 decibels from actual wind tunnel measurements. |
| **Mean Squared Error (MSE)** | **`22.59`** | Captures variance and penalizes larger outlier errors ($\text{RMSE} \approx 4.75\text{ dB}$). |
| **R-Squared ($R^2$)** | **`0.54`** | The linear baseline explains **54%** of the variance in acoustic noise across diverse aerodynamic regimes. |
| **Model Intercept ($\beta_0$)** | **`132.60 dB`** | Baseline acoustic level when standardized features are at zero. |

### 🔹 Sample Inference (Actual vs. Predicted)
After persisting the trained pipeline to disk (`PipelineModel`) and reloading it for production inference, the model generated the following predictions on the test set:
![Model Predictions Output](predictions_output.png)
### 🔹 Aerodynamic Insights from Model Coefficients
By inspecting the learned weights of the standardized Linear Regression model, we can extract meaningful physical insights into what drives airfoil noise:

| Aerodynamic Feature | Learned Coefficient | Physical & Engineering Impact |
| :--- | :--- | :--- |
| **`Frequency`** | `-3.9728` | **Strongest Inverse Driver:** Higher acoustic frequencies damp out faster and exhibit lower scaled sound pressure levels. |
| **`ChordLength`** | `-3.3818` | **Inverse Relationship:** Larger chord lengths shift the acoustic spectrum, reducing high-frequency trailing-edge noise. |
| **`AngleOfAttack`** | `-2.4775` | **Inverse Component (Coupled):** Interacts closely with boundary layer thickness (`SuctionSideDisplacement`) as flow separation changes. |
| **`SuctionSideDisplacement`** | `-1.6465` | **Boundary Layer Effect:** Reflects the damping and frequency-shifting behavior of thicker turbulent boundary layers. |
| **`FreeStreamVelocity`** | **`+1.5789`** | **Primary Positive Driver:** Faster airflow velocity directly increases kinetic energy and turbulence intensity, raising noise levels (dB). |

---
## 🚀 6. How to Run This Project
1. **Clone the repository:**
<pre><code>git clone https://github.com/YOUR_GITHUB_USERNAME/nasa-airfoil-noise-prediction-pyspark.git
cd nasa-airfoil-noise-prediction-pyspark</code></pre>
2. **Install dependencies:**
<pre><code>pip install pyspark==3.1.2 findspark</code></pre>
3. **Run the Jupyter Notebook or Python Script:**
   Open `Airfoil_Noise_Prediction_Pipeline.ipynb` in JupyterLab / Google Colab and execute the cells sequentially.
---
## 👨‍💻 Author & Contact
**Tariq Zeyad Jawad**  
*Physics Graduate | Data Engineering & Machine Learning Enthusiast*
* 🌐 **Website:** [tariqjawad.com](https://tariqjawad.com)
* 💼 **LinkedIn:** [linkedin.com/in/tariq-jawad](https://www.linkedin.com/in/tariq-jawad)
* 📧 **Email:** [tariq.z.jawad4@gmail.com](mailto:tariq.z.jawad4@gmail.com)
