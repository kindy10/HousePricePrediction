\# House Price Prediction



A machine learning project that predicts median house values using the California Housing dataset.



This project was built as part of my step-by-step journey into machine learning. The goal was not only to build a prediction model, but also to understand the complete machine learning workflow, including data exploration, visualization, model training, evaluation, hyperparameter experimentation, cross-validation, and model saving.



\## Project Goals



\- Understand the basic machine learning workflow

\- Explore and visualize a real-world dataset

\- Build a Linear Regression model

\- Build a Random Forest Regression model

\- Compare different machine learning models

\- Experiment with Random Forest hyperparameters

\- Use cross-validation

\- Train and save a final machine learning model

\- Practice Git and GitHub for version control



\## Dataset



This project uses the California Housing dataset.



The dataset contains information about houses and districts in California. The target variable is:



\- `MedHouseVal` - Median house value



The features used by the model are:



\- `MedInc` - Median income

\- `HouseAge` - Median house age

\- `AveRooms` - Average number of rooms

\- `AveBedrms` - Average number of bedrooms

\- `Population` - District population

\- `AveOccup` - Average number of occupants

\- `Latitude` - Latitude

\- `Longitude` - Longitude



\## Project Structure



```text

HousePricePrediction/

│

├── models/

│   └── house\_price\_random\_forest.pkl

│

├── notebooks/

│   └── house\_price\_prediction.ipynb

│

├── .gitignore

└── README.md

````



The trained model is stored locally in the `models/` directory. The `.pkl` file is excluded from Git because of its large size.



\## Machine Learning Workflow



The project follows these main steps:



\### 1. Load the Dataset



The California Housing dataset was loaded and inspected using Python and pandas.



\### 2. Explore the Data



I examined:



\* Number of rows and columns

\* Feature names

\* Data types

\* Missing values

\* Basic statistics

\* Target variable distribution



\### 3. Data Visualization



I used Matplotlib to visualize distributions and relationships in the dataset.



Examples include:



\* Histograms

\* Correlation analysis

\* Feature relationships



\### 4. Prepare the Data



The features were separated from the target:



```python

X = df.drop("MedHouseVal", axis=1)

y = df\["MedHouseVal"]

```



The dataset was then divided into training and testing sets.



\### 5. Linear Regression



I first trained a Linear Regression model as a baseline.



Results:



| Model             |      MAE |       R² |

| ----------------- | -------: | -------: |

| Linear Regression | 0.533200 | 0.575788 |



This provided a baseline for comparing more powerful models.



\### 6. Random Forest Regression



I then trained a Random Forest Regression model.



The Random Forest performed significantly better than Linear Regression.



Initial Random Forest results:



| Model         |      MAE |       R² |

| ------------- | -------: | -------: |

| Random Forest | 0.327731 | 0.804624 |



\## Hyperparameter Experiments



I experimented with several Random Forest hyperparameters.



\### `max\_depth`



The tested values were:



```text

5, 10, 15, 20, None

```



The best tested result was obtained with:



```text

max\_depth = None

```



\### `n\_estimators`



The tested values were:



```text

10, 50, 100, 200, 300

```



The best tested result was:



```text

n\_estimators = 300

```



with:



```text

MAE = 0.326662

R² = 0.806345

```



\### `max\_features`



I tested:



```text

1, 2, 3, 4, 5, 6, 7, 8

```



The best result was:



```text

max\_features = 3

```



with:



```text

MAE = 0.319261

R² = 0.817178

```



\### `min\_samples\_leaf`



I tested:



```text

1, 2, 4, 8, 16

```



The best result was:



```text

min\_samples\_leaf = 1

```



\### `min\_samples\_split`



I tested:



```text

2, 5, 10, 20, 40

```



The best result was:



```text

min\_samples\_split = 2

```



\## Cross-Validation



To check whether the model performed consistently across different subsets of the training data, I used 5-fold cross-validation.



Results:



```text

Cross-validation R² scores:

0.82086064

0.80888460

0.82168889

0.81646751

0.81635120

```



Mean cross-validation R²:



```text

0.816851

```



Standard deviation:



```text

0.004546

```



The relatively small standard deviation indicates that the model performed consistently across the five folds.



\## Final Model



The final Random Forest configuration was:



```python

RandomForestRegressor(

&#x20;   n\_estimators=300,

&#x20;   max\_depth=None,

&#x20;   max\_features=3,

&#x20;   min\_samples\_leaf=1,

&#x20;   min\_samples\_split=2,

&#x20;   random\_state=42

)

```



Final test performance:



```text

MAE ≈ 0.319

R²  ≈ 0.817

```



An R² of approximately 0.817 means that the model explains about 81.7% of the variance in the target variable on the test set.



\## Model Saving



The trained model was saved using `joblib`:



```python

joblib.dump(

&#x20;   final\_rf\_model,

&#x20;   "models/house\_price\_random\_forest.pkl"

)

```



The model file is excluded from Git because it is approximately 417 MB.



\## Technologies Used



\* Python

\* pandas

\* NumPy

\* Matplotlib

\* scikit-learn

\* joblib

\* Jupyter Notebook

\* Anaconda

\* Git

\* GitHub



\## What I Learned



Through this project I learned how to:



\* Load and inspect a real dataset

\* Work with pandas DataFrames

\* Analyze data statistically

\* Create visualizations

\* Separate features and target variables

\* Split data into training and testing sets

\* Train regression models

\* Evaluate models using MAE and R²

\* Understand Random Forests

\* Experiment with hyperparameters

\* Use cross-validation

\* Save trained machine learning models

\* Use Git branches, commits, and GitHub for version control



\## Future Improvements



Possible improvements for this project include:



\* More systematic hyperparameter tuning

\* Grid Search and Randomized Search

\* Additional regression models

\* Feature engineering

\* Better visualizations

\* Error analysis

\* A prediction interface

\* Model deployment as a web application



```

```



