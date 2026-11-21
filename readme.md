<img width="1845" height="1032" alt="image" src="https://github.com/user-attachments/assets/5343e397-83b2-4b32-93a9-3ad753e6984b" />

# Seoul Date Course Recommendation System

## 1. Introduction

This project is a machine learning-based recommendation system that combines location data from the Seoul area with users' personal preferences and rating histories to suggest an optimal four-stop date itinerary.

### Key Objectives and Challenges

- **Personalized Venue Rating Prediction:** By combining a user’s explicit preferences (such as interest in specific categories or preferred districts) with their historical rating data, the system calculates an estimated rating (on a 1–5 scale) for venues the user has not yet visited.

- **Hybrid Recommendation Modeling:** The approach integrates the strengths of both content-based filtering and collaborative filtering, utilizing a validation set to determine the optimal ensemble weights.

- **Addressing the "Cold Start" Problem:** For new users with no prior rating history, the system provides immediate recommendations by leveraging content-based predictions and proxy estimates derived from existing users with similar preferences.

- **Incorporating Practical Route Constraints:** Rather than simply listing four highly-rated venues, the system selects a route that maximizes the total predicted rating while adhering to specific constraints, including walking time between locations and total travel time.

**Gemini assisted with the line breaks and organization of the content in this document.**

---

## 2. System Architecture & Pipeline

The overall pipeline consists of two main stages:
1. **Predicting ratings for individual locations**
2. **Filtering and ranking 4-stop itineraries**

```text
[Input Data]
  ├── Location Metadata (places_for_recsys.csv)
  ├── User Profiles (users_synthetic.csv)
  ├── Rating History (ratings_train / validation / test.csv)
  └── Pool of Candidate 4-Stop Itineraries (four_stop_courses_estimated.csv)
         │
         ▼
[Stage 1: Calculating Predicted Ratings for Individual Locations]
  ├── Content-Based Model (Ridge Regression: category similarity + geographic proximity + user rating leniency)
  └── Collaborative Filtering Model (Weighted ensemble of Item-based KNNWithMeans + SVD++)
         │
         ▼  (Determining optimal weight β based on validation RMSE and combining models)
[Stage 2: Itinerary Filtering and Final Ranking]
  ├── Excluding previously visited (rated) locations and removing duplicate stops within itineraries
  ├── Applying travel time constraints (≤ 40 minutes of walking between stops; ≤ 75 minutes of total walking time)
  └── Sorting by the sum of predicted ratings (descending) and total travel time (ascending)
```

### Key Module Components

| Filename | Role and Main Functions |
| :--- | :--- |
| **`contents_rec.py`** | Core module for content-based filtering. Handles feature engineering, comparison of model variants (A–F), prevention of data leakage (via Leave-One-Out), tie-breaking, Top-K evaluation, and provides a function (`recommend_for_new_user`) for generating recommendations for new users. |
| **`final_course_pipeline.py`** | Integrated pipeline for existing users. Combines content-based filtering (Ridge) with collaborative filtering (KNNWithMeans, SVD++) using weighted hybrid scoring to generate an optimal 4-location walking course for a specific user (`--user-id`). |
| **`new_user_both_models.py`** | Interactive hybrid recommendation system for new users. Incorporates user-provided category preferences and preferred districts; combines content-based scores with proxy scores derived from similar users (collaborative filtering) to suggest courses through a step-by-step geographic expansion process. |

---

## 3. Dataset & Preprocessing

### Input Data Specifications

- **`places_for_recsys.csv`**: Data on 1,047 locations in Seoul (`place_id`, `name`, `category`, `district`, `latitude`, `longitude`, `address`)

- **`users_synthetic.csv`**: Profiles of 2,000 synthetic users (`user_id`, `interest_attractions`, `interest_culture`, `interest_shopping`, `preferred_district`, `rating_generosity`)

- **`ratings_train.csv` / `ratings_validation.csv` / `ratings_test.csv`**: User-location rating data (on a scale of 1–5)

- **`four_stop_courses_estimated.csv`**: Pre-generated candidate itineraries consisting of 4 locations, including estimated walking times for each segment (`leg1_min`, `leg2_min`, `leg3_min`, `total_estimated_minutes`)

### Preprocessing and Feature Engineering

1. **Handling Missing Values:** Missing values for districts (`district`, `preferred_district`) are replaced with the label `"Unknown"` rather than being removed. This ensures that these entries remain in the pool of potential recommendations; however, they are excluded from the one-hot encoding process to prevent false matches between "Unknown" entries.

2. **Multi-Block One-Hot Vectorization:** Categories are structured into a 3-dimensional space (covering "Attractions," "Culture," and "Shopping"), while districts are represented as a 25-dimensional space. This allows for the independent calculation of cosine similarity for each feature block.

3. **Exponential Decay-Based Regional Proximity (`exp_near`):** To mitigate the disadvantage faced by locations near district boundaries—which occurs when proximity is determined solely by whether two places belong to the same district—we calculate an exponential decay-based proximity score centered on the median coordinates of each district.

   $$\text{Proximity} = \exp\left(-\frac{d_{\text{km}}}{s_{\text{km}}}\right)$$

   *(Through experiments on the validation set, the optimal reference distance $s_{\text{km}}$ was determined to be 3 km.)*

---

## 4. Recommendation Methodology (Recommendation Modeling and Course Optimization)

### 4.1 Content-Based Filtering (`contents_rec.py`)

- **Explicit & Learned Profiles:** In addition to a user’s initial interest vector, we constructed a "learned" profile by incorporating the user’s past rating deviations ($r_{u,i} - \bar{r}_u$) as weights. To prevent data leakage—where the target row might otherwise be included in the feature set—we applied a **Leave-One-Out (LOO)** approach during training.

- **Model Variants (A–F) and Ridge Regression:** We trained `Ridge` regression models for various feature combinations (Models A–F). Based on validation RMSE, we selected the optimal feature set (`exp_cat`, `exp_near`, `generosity`) and the best regularization coefficient ($\alpha$).

### 4.2 Collaborative Filtering and Hybrid Ensemble (`final_course_pipeline.py`)

- **Internal CF Ensemble:** We computed a weighted average of predictions from an item-based `KNNWithMeans` model (using cosine similarity, $k=40$) and a latent factor model, `SVD++` ($n_{\text{factors}}=40$):

  $$\alpha_{\text{cf}} \cdot \hat{r}_{\text{KNN}} + (1 - \alpha_{\text{cf}}) \cdot \hat{r}_{\text{SVD++}}$$

- **Final Hybrid Integration:** Through a grid search (with a step size of 0.05) on the validation set, we determined the optimal mixing ratio, $\beta$, for combining the content-based predictions ($\hat{r}_{\text{CB}}$) and the collaborative filtering predictions ($\hat{r}_{\text{CF}}$).

  $$\hat{r}_{\text{final}} = \beta \cdot \hat{r}_{\text{CB}} + (1 - \beta) \cdot \hat{r}_{\text{CF}}$$

### 4.3 Handling the Cold Start Problem for New Users (`new_user_both_models.py`)

- **Proxy Estimation for Collaborative Filtering (CF):** Instead of assigning an arbitrary existing user ID to a new user, the system identifies the top $k$ most similar existing users based on a weighted combination of category preference similarity (80%) and whether they reside in the same administrative district (20%). It then calculates a CF prediction by taking a similarity-weighted average of the ratings from these users.

- **Tiered Search Based on Geographic Priority:** If it is not possible to find a 4-location course within the selected district that meets the travel constraints, the search scope is automatically expanded in stages:
  1. Within the selected district
  2. Including the 5 adjacent districts (relative to a central point)
  3. The entirety of Seoul

### 4.4 Criteria for Selecting a 4-Location Date Course

- **Filtering Constraints:** All 4 locations must be distinct; previously rated locations are excluded; the walking time between each consecutive location must be $\le 40$ minutes; and the total walking time for the entire course must be $\le 75$ minutes.

- **Optimization Objective Function:** Among courses that satisfy the constraints, the system aims to maximize the sum of the predicted ratings (`rating_sum`) for the 4 locations. In the event of a tie, the course with the shorter total estimated walking time (`total_estimated_minutes`) is prioritized.

---

## 5. Evaluation & Results

To objectively verify the model's performance, we evaluate both rating prediction accuracy and Top-K ranking quality from multiple perspectives.

*(For the final evaluation, the model is retrained on the combined `Train + Validation` dataset, and then tested exactly once on the `Test` set.)*

### Rating Prediction Metrics (`MAE`, `RMSE`)

<img width="1920" height="1260" alt="metric_mae" src="https://github.com/user-attachments/assets/f259f22d-3f22-491d-a43f-fb5226d4a1e3" />

<img width="1920" height="1260" alt="metric_rmse" src="https://github.com/user-attachments/assets/63b31cf2-cd36-455f-967c-406222645fd8" />

### Ranking and Recommendation Diversity Metrics (`Precision@10`, `Recall@10`, `NDCG@10`)

<img width="1920" height="1260" alt="metric_precision_at_10" src="https://github.com/user-attachments/assets/702f10ac-059f-478c-b91f-c888e11a9078" />

<img width="1920" height="1260" alt="metric_recall_at_10" src="https://github.com/user-attachments/assets/8b34c1f1-5da2-419d-9bb3-09034c755b4d" />

<img width="1920" height="1260" alt="metric_ndcg_at_10" src="https://github.com/user-attachments/assets/b795ef4c-0732-43bd-b552-71fc626339d9" />

### Evaluation Metric Insights

- **Rating Prediction Accuracy:** Blending **Content Ridge** and **CF KNN + SVD++** into the **Combined** model reduces both `RMSE` (`0.7195`) and `MAE` (`0.5816`) compared to either single model, showing complementary benefits between content/proximity features and collaborative filtering.

- **Top-K Ranking Trade-off:** **Content Ridge** achieves the highest `NDCG@3`, `NDCG@10`, and `Precision@10`, while **Combined** closely follows and reaches the best `Recall@10` (`0.035360`).

- **Why Top-K Values Are Small:** Ranking metrics are relatively low because the models optimize rating prediction rather than listwise ranking, there is a large candidate pool (`1,047` places) with sparse relevant items (`rating >= 4`) per user, and `KNNWithMeans` scores in top-K evaluation use a fast matrix approximation.

- **Tie-Break Optimization:** In cases where predicted scores are identical, `np.lexsort` is used to apply secondary criteria—specifically, sorting by average rating or the number of ratings—to optimize the `NDCG@10` metric.

---

## 6. Project Structure (Folder and File Organization)

```text
project_root/
├── contents_rec.py                 # Content-based recommendation: training, evaluation, and score table generation
├── final_course_pipeline.py        # Hybrid recommendation for existing users and selection of 4-location courses
├── new_user_both_models.py         # Interactive recommendation for new users (combining content-based and proxy collaborative filtering)
│
├── 02_Content-based/
│   ├── places_for_recsys.csv       # Metadata for locations in Seoul
│   ├── users_synthetic.csv         # Synthetic user profile data
│   ├── ratings_validation.csv      # Rating data for validation
│   ├── ratings_test.csv            # Rating data for testing
│   └── output/                     # Results from running contents_rec.py (score tables, comparison tables, metrics.json)
│
├── 03_CF_Filtering/
│   └── ratings_train.csv           # Rating data for training
│
└── four_stop_courses_estimated.csv # Candidate 4-location courses and estimated travel time data
```

---

## 7. Getting Started / Usage

### Environment Setup and Required Packages

Python 3.8 or higher is recommended. The following libraries are required:

```bash
pip install numpy pandas scikit-learn scikit-surprise
```

### How to Run

#### 1. Train and compare content-based models, and generate a comprehensive score table

```bash
python3 contents_rec.py
```

- Results will be saved in the `02_콘텐츠기반_추천/output/` directory in CSV and JSON formats.

#### 2. Evaluate the hybrid model for existing users and recommend 4 date course locations

```bash
python3 final_course_pipeline.py --data-root . --user-id U0001 --output-dir final_course_output
```

- `--user-id`: The ID of an existing user for whom recommendations will be generated (default: `U0001`).

#### 3. Interactive course recommendations for new users (hybrid model version)

```bash
python3 new_user_both_models.py --data-root . --neighbors 20 --output-dir new_user_both_output
```

- After running, follow the on-screen prompts in the terminal to enter your preferences (0–10) for `Attractions / Culture / Shopping` and your `preferred district`.

---

## 8. Limitations & Future Work

- **Limitations of Walking Time Estimates:** The current walking time estimates for candidate routes (`leg1_min` through `leg3_min`) are based on straight-line distance adjustments rather than verified data from actual road networks. Future iterations could improve the accuracy of these routes by integrating APIs for pedestrian navigation, such as those provided by Kakao Mobility, TMAP, or OSRM.

- **Training Based on Synthetic User Data:** As the model was validated using `users_synthetic.csv` and synthetic rating distributions, it will be necessary to collect real-world service logs—such as user clicks, dwell times, and actual visit reviews—to recalibrate the model's weights through online A/B testing.

- **Incorporating Operating Hours and Dynamic Constraints:** Recommendation quality could be further enhanced by integrating operational details—such as business hours and days off—as well as logical sequencing rules for different categories (e.g., prioritizing a flow of "attractions $\rightarrow$ dining $\rightarrow$ cafés") during the route-filtering stage.
