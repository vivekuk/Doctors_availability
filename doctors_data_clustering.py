# Create a synthetic dataset with multiple features, including categorical columns
excel_file_path = "doctors_data.xlsx"

df = pd.read_excel(excel_file_path)

# Display the original data
print("Original Data:")
print(df.head())

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Use LabelEncoder to convert categorical columns to numerical values
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Select features for clustering
features_for_clustering = df[["Doctor Name", "Speciality", "Education", "Registration No.", "Experience", "Location"]]

# Standardize numerical features
numeric_cols = ["Registration No.", "Experience"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols)
    ])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)

# Create a pipeline for preprocessing and clustering
kmeans_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', kmeans)
])

# Apply k-modes clustering
kmodes = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1, random_state=42)

# Create a pipeline for preprocessing and clustering
kmodes_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmodes', kmodes)
])

# Fit and predict
df['kmeans_Cluster'] = kmeans_pipeline.fit_predict(features_for_clustering)
df['kmodes_Cluster'] = kmodes_pipeline.fit_predict(features_for_clustering)

# Display the clustered data
print("\nClustered Data:")
print(df.head())
