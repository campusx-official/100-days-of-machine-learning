# Types of Machine Learning

Machine Learning is broadly divided into **4 main types** based on how the model learns from data:

```
Machine Learning
├── 1. Supervised Learning
│       ├── Regression
│       └── Classification
├── 2. Unsupervised Learning
│       ├── Clustering
│       ├── Dimensionality Reduction
│       ├── Anomaly Detection
│       └── Association Rule Learning
├── 3. Semi-Supervised Learning
└── 4. Reinforcement Learning
```

---

---

## 1. 🟦 Supervised Learning

### What is it?
The model is trained on **labeled data** — meaning every input has a corresponding **correct output (label)**.  
The model learns the mapping: `Input (X) → Output (Y)`

> **Analogy:** A student learning from a textbook that has **questions AND answers**. The student studies examples and learns to answer new questions.

### How it works:
1. Feed the model (input, label) pairs
2. Model makes a prediction
3. Compare prediction to the correct label using a **Loss Function**
4. Model adjusts itself using **Gradient Descent** to reduce the error
5. Repeat until the model learns well

---

### 🔹 A) Regression

**Goal:** Predict a **continuous numerical value**

| Algorithm | Description |
|---|---|
| Linear Regression | Fits a straight line to data |
| Polynomial Regression | Fits a curved line |
| Ridge / Lasso Regression | Linear regression with regularization |
| SVR (Support Vector Regression) | SVM applied to regression |
| Random Forest Regressor | Ensemble of decision trees for regression |

#### Real-Life Examples of Regression:

| Use Case | Input Features | Output (Prediction) |
|---|---|---|
| 🏠 House Price Prediction | Area, Location, Rooms, Age | Price in ₹ or $ |
| 🌡️ Weather Forecasting | Humidity, Pressure, Wind Speed | Temperature (°C) |
| 📈 Stock Price Prediction | Historical prices, Volume | Future price |
| 🚗 Car Price Estimation | Brand, Age, Mileage, Fuel type | Selling price |
| 🏥 Medical Cost Prediction | Age, BMI, Smoker, Region | Insurance cost |
| ⚡ Electricity Demand | Time of day, Season, Temperature | Units consumed |

---

### 🔹 B) Classification

**Goal:** Predict a **category/class label** from a fixed set of classes

| Algorithm | Description |
|---|---|
| Logistic Regression | Binary classification using sigmoid function |
| Decision Tree | Tree-based rule classification |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | Finds best boundary between classes |
| K-Nearest Neighbors (KNN) | Classifies based on nearest neighbors |
| Naive Bayes | Probabilistic classifier (great for text) |
| Neural Networks | Deep learning-based classification |

#### Types of Classification:
- **Binary Classification** — 2 classes (spam/not spam, disease/no disease)
- **Multi-class Classification** — more than 2 classes (cat/dog/bird)
- **Multi-label Classification** — multiple labels at once (a photo tagged: beach + sunset + vacation)

#### Real-Life Examples of Classification:

| Use Case | Input Features | Output (Class) |
|---|---|---|
| 📧 Email Spam Detection | Email text, sender, links | Spam / Not Spam |
| 🏥 Disease Diagnosis | Symptoms, test results, age | Disease / No Disease |
| 😊 Sentiment Analysis | Product review text | Positive / Negative / Neutral |
| 📸 Image Recognition | Pixel values of image | Cat / Dog / Car / etc. |
| 💳 Credit Card Fraud Detection | Transaction amount, location, time | Fraud / Not Fraud |
| 🎵 Music Genre Classification | Audio features (tempo, beat) | Rock / Pop / Jazz |
| 📰 News Categorization | Article text | Sports / Politics / Tech |
| 🔐 Face Recognition | Facial features from image | Person A / Person B / Unknown |

---

### ✅ When to use Supervised Learning?
- You have **labeled data** available
- You want to **predict** a specific output
- Problem is well-defined (regression or classification)

---

---

## 2. 🟩 Unsupervised Learning

### What is it?
The model is trained on **unlabeled data** — no correct answers are given.  
The model must **discover hidden patterns, structure, or groupings** on its own.

> **Analogy:** A child placed in a room full of toys — no one tells them which is which. They group similar toys together on their own (cars together, dolls together).

---

### 🔹 A) Clustering

**Goal:** Group similar data points together into **clusters** without predefined labels

| Algorithm | Description |
|---|---|
| K-Means | Divides data into K clusters based on distance |
| DBSCAN | Finds clusters of arbitrary shape, handles outliers |
| Hierarchical Clustering | Builds a tree (dendrogram) of clusters |
| Gaussian Mixture Models (GMM) | Soft probabilistic clustering |
| Mean Shift | Finds clusters by shifting toward dense regions |

#### Real-Life Examples of Clustering:

| Use Case | Description |
|---|---|
| 🛍️ Customer Segmentation | Group customers into: Budget shoppers, Premium buyers, Occasional buyers — for targeted marketing |
| 📰 News Article Grouping | Google News groups similar articles from different sources into one story |
| 🧬 Gene Expression Analysis | Group genes with similar expression patterns to study diseases |
| 🗺️ City Zone Planning | Group areas by population density, income, infrastructure for urban planning |
| 🖼️ Image Segmentation | Group pixels of similar color/texture (used in photo editing, medical imaging) |
| 🎬 Movie/Music Recommendation | Cluster users with similar taste, then recommend what similar users liked |
| 🌍 Social Network Analysis | Detect communities/groups in a social network |

---

### 🔹 B) Dimensionality Reduction

**Goal:** Reduce the number of features (columns) in data while **preserving important information**

> **Why?** High-dimensional data (hundreds of features) causes:
> - Slow training
> - Overfitting
> - Hard to visualize
> - "Curse of Dimensionality"

| Algorithm | Description |
|---|---|
| PCA (Principal Component Analysis) | Projects data onto directions of maximum variance |
| t-SNE | 2D/3D visualization of high-dimensional data |
| UMAP | Faster alternative to t-SNE for visualization |
| Autoencoders | Neural network that compresses and reconstructs data |
| LDA (Linear Discriminant Analysis) | Supervised dimensionality reduction |
| SVD (Singular Value Decomposition) | Matrix factorization for compression |

#### Real-Life Examples of Dimensionality Reduction:

| Use Case | Description |
|---|---|
| 👤 Face Recognition | A face image has thousands of pixels → PCA reduces to key facial features |
| 🧬 Genomics | DNA data has millions of features → reduce to key genes for analysis |
| 📊 Data Visualization | Plot 100-feature customer data in 2D to find patterns visually (t-SNE) |
| 🗣️ NLP / Text Data | Document-term matrix has thousands of words → compress to topics |
| 🖼️ Image Compression | JPEG uses SVD-like compression to reduce file size |
| 🏥 Medical Imaging | MRI scans reduced to key features for disease classification |
| 📉 Noise Removal | Remove irrelevant/noisy features from sensor data |

---

### 🔹 C) Anomaly Detection (Outlier Detection)

**Goal:** Identify **unusual data points** that don't fit the normal pattern

> **Analogy:** A security guard watching CCTV — most people walk normally, but someone running or behaving unusually is flagged.

| Algorithm | Description |
|---|---|
| Isolation Forest | Isolates anomalies by random splits — anomalies are easier to isolate |
| One-Class SVM | Learns boundary of normal data, flags everything outside |
| Autoencoders | Reconstruction error is high for anomalies |
| DBSCAN | Points not belonging to any cluster = anomalies |
| Z-Score / IQR | Statistical methods for simple anomaly detection |
| LOF (Local Outlier Factor) | Compares local density of a point to its neighbors |

#### Real-Life Examples of Anomaly Detection:

| Use Case | Description |
|---|---|
| 💳 Credit Card Fraud | A transaction in New York followed by one in Tokyo 2 mins later = anomaly |
| 🏭 Manufacturing Defects | Sensors detect unusual machine vibration → predict equipment failure |
| 🖥️ Network Intrusion Detection | Unusual traffic patterns → possible cyberattack |
| 🏥 Patient Health Monitoring | Abnormal heart rate or blood pressure → emergency alert |
| 📦 Supply Chain | Unusual spike or drop in inventory → flag for review |
| 🌐 Web Traffic | Sudden traffic spike from one IP → possible bot attack |
| ✈️ Aircraft Maintenance | Sensor readings outside normal range → maintenance required |

---

### 🔹 D) Association Rule Learning

**Goal:** Find **interesting relationships / rules** between variables in large datasets

> **Classic Example:** "Customers who buy bread and butter also tend to buy jam"  
> Rule: `{bread, butter} → {jam}` with high confidence

#### Key Terms:
| Term | Meaning |
|---|---|
| **Support** | How frequently the itemset appears in the dataset |
| **Confidence** | How often the rule is correct (if A, then B) |
| **Lift** | How much more likely B is given A, compared to random chance |

| Algorithm | Description |
|---|---|
| Apriori | Finds frequent itemsets level by level |
| FP-Growth | Faster — uses a tree structure (FP-tree) |
| Eclat | Uses vertical data format, faster for dense data |

#### Real-Life Examples of Association Rule Learning:

| Use Case | Description |
|---|---|
| 🛒 Market Basket Analysis | Walmart discovered: people who buy diapers on Fridays also buy beer → placed them near each other |
| 🎬 Movie Recommendations | Users who watched Movie A & B also watched Movie C → recommend Movie C |
| 🛍️ E-commerce (Amazon) | "Customers who bought this also bought..." feature |
| 💊 Medical Diagnosis | Patients with symptom A & B are often diagnosed with Disease C |
| 📚 Library / Book Recommendation | Books frequently borrowed together are recommended together |
| 🌐 Web Usage Mining | Users who visit Page A often visit Page B next → optimize navigation |
| 🍕 Menu Engineering | Restaurant finds: pizza + garlic bread is frequently ordered together → create combo deal |

---

### ✅ When to use Unsupervised Learning?
- No labeled data available
- Want to **explore and understand** the data
- Goal is **pattern discovery**, not prediction

---

---

## 3. 🟨 Semi-Supervised Learning

### What is it?
Uses a **small amount of labeled data** + a **large amount of unlabeled data** for training.

> **Why?** Labeling data is **expensive and time-consuming** (requires human experts).  
> Semi-supervised learning gets the best of both worlds.

> **Analogy:** A student reads 10 solved math problems (labeled) and then practices 1000 unsolved problems (unlabeled) on their own — learning patterns from both.

### How it works:
1. Train initial model on small labeled dataset
2. Use model to **pseudo-label** the unlabeled data (self-training)
3. Retrain on labeled + pseudo-labeled data
4. Repeat to improve

### Techniques:
| Technique | Description |
|---|---|
| Self-Training | Model labels unlabeled data and retrains on it |
| Co-Training | Two models train on different views of data and label for each other |
| Label Propagation | Propagates labels through a graph of similar data points |
| Generative Models | VAE, GANs used to learn data distribution from unlabeled data |

### Real-Life Examples of Semi-Supervised Learning:

| Use Case | Description |
|---|---|
| 🏥 Medical Image Labeling | Only 500 MRI scans labeled by doctors (expensive) + 50,000 unlabeled scans → semi-supervised model for tumor detection |
| 😊 Sentiment Analysis | 1,000 labeled reviews + millions of unlabeled reviews → better sentiment classifier |
| 🎙️ Speech Recognition | Small labeled audio + large unlabeled audio data → Google/Apple voice assistants |
| 🌐 Web Content Classification | Few labeled web pages + billions of unlabeled pages → categorize the web |
| 📸 Image Classification | Google Photos: few labeled images + many unlabeled → identify people, places |
| 🧬 Protein Structure Prediction | Few known protein structures + millions of unknown sequences (AlphaFold used this idea) |

### ✅ When to use Semi-Supervised Learning?
- Labeled data is **scarce or expensive** to obtain
- Large amounts of **unlabeled data** are available
- You want **better accuracy** than purely unsupervised, without full labeling cost

---

---

## 4. 🟥 Reinforcement Learning (RL)

### What is it?
An **agent** learns to make decisions by **interacting with an environment**.  
It receives **rewards** for good actions and **penalties** for bad ones — and learns to maximize total reward over time.

> **Analogy:** Training a dog 🐕 — give it a treat (reward) when it does the right thing, say "No!" (penalty) when it does wrong. The dog learns to repeat rewarded behavior.

### Key Concepts:
| Term | Meaning |
|---|---|
| **Agent** | The learner / decision maker (e.g., robot, game AI) |
| **Environment** | The world the agent interacts with |
| **State (S)** | Current situation of the agent |
| **Action (A)** | What the agent can do |
| **Reward (R)** | Feedback signal (+ve for good, -ve for bad action) |
| **Policy (π)** | Strategy the agent uses to choose actions |
| **Value Function** | Expected future reward from a given state |
| **Episode** | One complete run of the agent in the environment |

### RL Loop:
```
Agent observes State (S)
    → Takes Action (A)
    → Environment gives Reward (R) + New State (S')
    → Agent updates its Policy
    → Repeat until goal is achieved
```

### Key Algorithms:
| Algorithm | Description |
|---|---|
| Q-Learning | Learn value of actions using a Q-table |
| Deep Q-Network (DQN) | Q-Learning with a deep neural network (used by DeepMind) |
| SARSA | On-policy version of Q-Learning |
| Policy Gradient (REINFORCE) | Directly optimize the policy |
| PPO (Proximal Policy Optimization) | State-of-the-art RL algorithm (used by ChatGPT/RLHF) |
| Actor-Critic (A3C, A2C) | Combines policy gradient + value function |
| AlphaZero / MuZero | Self-play RL for games (chess, Go) |

### Real-Life Examples of Reinforcement Learning:

| Use Case | Description |
|---|---|
| 🎮 Game Playing | DeepMind's AlphaGo beat world champion at Go; OpenAI Five beat Dota 2 pros |
| 🤖 Robotics | Robot learns to walk, grasp objects, or assemble parts by trial and error |
| 🚗 Self-Driving Cars | Car learns to navigate, avoid obstacles, follow traffic rules (Tesla, Waymo) |
| 💬 ChatGPT / LLMs | RLHF (RL from Human Feedback) — model learns to give better answers based on human ratings |
| 📈 Algorithmic Trading | Trading agent learns when to buy/sell stocks to maximize profit |
| ⚡ Data Center Cooling | Google used RL to reduce data center cooling energy by 40% |
| 🏥 Drug Discovery | RL agent designs molecules that optimize for drug effectiveness |
| 🎯 Ad Recommendation | Which ad to show to maximize click-through rate |
| ✈️ Flight Control | RL agents control aircraft autopilot systems |
| 🕹️ Video Game NPC AI | NPCs in games learn to play more intelligently against the player |

### ✅ When to use Reinforcement Learning?
- Task involves **sequential decision making**
- A **reward signal** can be defined
- No labeled data — learning happens through **trial and error**
- Environment can be **simulated** for training (games, robotics simulators)

---

---

## 📊 Final Comparison — All 4 Types

| Feature | Supervised | Unsupervised | Semi-Supervised | Reinforcement |
|---|---|---|---|---|
| **Data Required** | Labeled | Unlabeled | Labeled + Unlabeled | No dataset needed |
| **Goal** | Predict output | Find patterns | Predict with less labels | Learn optimal behavior |
| **Feedback** | Correct labels | None | Partial labels | Reward / Penalty |
| **Output** | Prediction | Clusters / Structure | Prediction | Policy / Actions |
| **Examples** | Spam filter, Price prediction | Customer segmentation | Medical imaging | Game AI, Robotics |
| **Algorithms** | SVM, Random Forest, Neural Net | K-Means, PCA, Apriori | Self-training, Label Propagation | Q-Learning, PPO, DQN |
| **Data labeling cost** | High | None | Low | None |

---

## 🧠 Quick Memory Aid

```
SUPERVISED   → Teacher gives correct answers → Learn to predict
UNSUPERVISED → No teacher → Discover hidden patterns
SEMI-SUPER   → Few answers + lots of data → Best of both worlds
REINFORCEMENT → Learn by doing → Trial, Error, Reward
```

---

*Next Topic →* Supervised Learning in Depth — Regression Algorithms
