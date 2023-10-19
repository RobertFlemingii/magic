What is Machine Learning?
Machine Learning is a subdomain of computer science that focuses on algorithms which help a computer learn from data without explicit programming

AI vs ML vs DS
Artificial Intelligence is an area of computer science, where the goal is to enable computers and machines to perform human-like tasks and simulate human behavior
Machine learning is a subset of AI that tries to solve a specific problem and make predictions using data
Data science is a field that attempts to find patterns and draw insights from data (might use ML!)
All fields overlap! All may use ML!

Types of Machine Learning
01  Supervised learning - uses labeled inputs (meaning the input has a corresponding output label) to train models and learn outputs
02 Unsupervised learning - uses unlabeled data to learn about patterns in data
03 Reinforcement learning - agent learning in interactive environment based on rewards and penalties

Features
o Qualitative - categorical data (finite number of categories or groups)
NOMINAL DATA (no inherent order)
ORDINAL DATA (inherent order)
o Quantitative - numerical valued data (could be discrete or continuous)

Supervised Learning Tasks
01 Classification - predict discrete classes
MULTICLASS CLASSIFICATION 
BINARY CLASSIFICATION
02 Regression - predict continuous values

Loss Functions
L1 Loss     loss = sum(|y_real - y_predicted|)
L2 Loss     loss = sum((y_real - y_predicted)^2)
Binary Cross-Entropy Loss   loss = -1 / N * sum(y_real * log(y_predicted) + (1 - y_real) * log((1-y_predicted)))
(You just need to know that loss decreases as the performance gets better)

