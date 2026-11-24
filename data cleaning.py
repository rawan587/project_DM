
import pandas as pd



df = pd.read_csv("StudentPerformanceFactors.csv")

print("Number of rows and columns:", df.shape)
print(df.head())



print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Summary of the number columns ---")
print(df.describe())



print("\nDuplicate rows before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after:", df.duplicated().sum())



print("\n--- Missing values per column ---")
print(df.isnull().sum())

# Only 3 text columns have missing values.
# We fill each one with its most common value (the mode).
df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])
df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])
df["Distance_from_Home"] = df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])

print("Missing values after filling:", df.isnull().sum().sum())




# Fix Hours_Studied
q1 = df["Hours_Studied"].quantile(0.25)
q3 = df["Hours_Studied"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("\nHours_Studied outliers:", ((df["Hours_Studied"] < lower) | (df["Hours_Studied"] > upper)).sum())
# clip = replace values outside the limits with the limit itself (we do NOT delete rows)
df["Hours_Studied"] = df["Hours_Studied"].clip(lower, upper)

# Fix Tutoring_Sessions
q1 = df["Tutoring_Sessions"].quantile(0.25)
q3 = df["Tutoring_Sessions"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("Tutoring_Sessions outliers:", ((df["Tutoring_Sessions"] < lower) | (df["Tutoring_Sessions"] > upper)).sum())
df["Tutoring_Sessions"] = df["Tutoring_Sessions"].clip(lower, upper)

# Exam_Score: a score cannot be more than 100, so we cap it at 100
print("Exam_Score values above 100:", (df["Exam_Score"] > 100).sum())
df["Exam_Score"] = df["Exam_Score"].clip(upper=100)



df["Parental_Involvement"] = df["Parental_Involvement"].map({"Low": 0, "Medium": 1, "High": 2})
df["Access_to_Resources"] = df["Access_to_Resources"].map({"Low": 0, "Medium": 1, "High": 2})
df["Motivation_Level"] = df["Motivation_Level"].map({"Low": 0, "Medium": 1, "High": 2})
df["Family_Income"] = df["Family_Income"].map({"Low": 0, "Medium": 1, "High": 2})
df["Teacher_Quality"] = df["Teacher_Quality"].map({"Low": 0, "Medium": 1, "High": 2})
df["Peer_Influence"] = df["Peer_Influence"].map({"Negative": 0, "Neutral": 1, "Positive": 2})
df["Parental_Education_Level"] = df["Parental_Education_Level"].map({"High School": 0, "College": 1, "Postgraduate": 2})
df["Distance_from_Home"] = df["Distance_from_Home"].map({"Near": 0, "Moderate": 1, "Far": 2})

# These columns have only 2 values, so we use 0 and 1
df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
df["Internet_Access"] = df["Internet_Access"].map({"No": 0, "Yes": 1})
df["Learning_Disabilities"] = df["Learning_Disabilities"].map({"No": 0, "Yes": 1})
df["School_Type"] = df["School_Type"].map({"Public": 0, "Private": 1})
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

print("\n--- Data types after encoding (should all be numbers) ---")
print(df.dtypes)
print("Missing values after encoding:", df.isnull().sum().sum())



print("\nExam_Score mean:", round(df["Exam_Score"].mean(), 2))
print("Exam_Score min:", df["Exam_Score"].min(), "| max:", df["Exam_Score"].max())


correlation = df.corr()["Exam_Score"].drop("Exam_Score").sort_values()
print("\n--- Correlation with Exam_Score ---")
print(correlation)



df.to_csv("student_cleaned_encoded.csv", index=False)
print("\nSaved student_cleaned_encoded.csv, shape:", df.shape)

