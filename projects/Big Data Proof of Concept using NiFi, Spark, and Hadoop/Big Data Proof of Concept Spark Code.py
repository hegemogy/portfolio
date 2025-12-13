'''
This was developed from a provided template, but the data handling and model training/evaluation
were implemented specifically by me for the mental health and social media dataset.
'''
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime
import happybase

# Step 1: Create Spark session
spark = (SparkSession.builder
         .appName("MentalHealthSocialMediaML")
         .enableHiveSupport()
         .getOrCreate())

# Step 2: Load Hive table into Spark DataFrame
df = spark.sql(
    "SELECT User_ID, Age, Gender, Daily_Screen_Time, Sleep_Quality, "
    "Stress_Level, Days_Without_Social_Media, Exercise_Frequency, "
    "Social_Media_Platform, Happiness_Index FROM user_social_media_data"
)

# Step 3: Handle null values
df = df.na.drop()

# Step 4: Encode categorical variables
gender_indexer = StringIndexer(
    inputCol="Gender", outputCol="GenderIndex"
)
platform_indexer = StringIndexer(
    inputCol="Social_Media_Platform", outputCol="PlatformIndex"
)

df = gender_indexer.fit(df).transform(df)
df = platform_indexer.fit(df).transform(df)

# Step 5: Create binary label (High Happiness = 1 if >=7, else 0)
df = df.withColumn("label", (df["Happiness_Index"] >= 7).cast("int"))

# Step 6: Assemble features
feature_cols = [
    "Age", "Daily_Screen_Time", "Sleep_Quality", "Stress_Level",
    "Days_Without_Social_Media", "Exercise_Frequency",
    "GenderIndex", "PlatformIndex"
]

assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features"
)
assembled_df = assembler.transform(df).select("features", "label")

# Step 7: Split into train/test
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)
train_rows = train_data.count()
test_rows = test_data.count()

# Step 8: Train Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Step 9: Evaluate model
predictions = lr_model.transform(test_data)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="precisionByLabel"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="recallByLabel"
)

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# For binary classification, label 1 = "High Happiness"
precision = evaluator_precision.evaluate(
    predictions, {evaluator_precision.metricLabel: 1}
)
recall = evaluator_recall.evaluate(
    predictions, {evaluator_recall.metricLabel: 1}
)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print(f"Precision (High Happiness): {precision}")
print(f"Recall (High Happiness): {recall}")

row_key = "logreg_01"
model_info = {
    "algorithm": "LogisticRegression",
    "features_used": ",".join(feature_cols),
    "hyperparameters": f"maxIter={lr.getMaxIter()}, "
                       f"regParam={lr.getRegParam()}",
    "run_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "train_rows": str(train_rows),
    "test_rows": str(test_rows)
}
evaluation = {
    "accuracy": accuracy,
    "f1_score": f1_score,
    "precision": precision,
    "recall": recall
}

# ---- Step 10: Write metrics to HBase with HappyBase ----
data = []
for k, v in model_info.items():
    data.append((row_key, f"model_info:{k}", str(v)))
for k, v in evaluation.items():
    data.append((row_key, f"evaluation:{k}", str(v)))

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('ml_metrics')
    for row in partition:
        row_key, column, value = row
        table.put(row_key.encode(),
                  {column.encode(): value.encode()})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 11: Stop Spark session
spark.stop()