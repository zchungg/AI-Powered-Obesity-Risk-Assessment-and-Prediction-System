# Imports Path to create and manage file and folder paths across different operating systems.
from pathlib import Path

# Gets the project's root directory by locating this file and moving up two directory levels.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Defines the folder where the dataset is stored.
DATA_DIR = ROOT_DIR / "data"

# Defines the folder where trained machine learning models are stored.
MODEL_DIR = ROOT_DIR / "models"

# Defines the folder where model evaluation results are stored.
RESULTS_DIR = ROOT_DIR / "results"


# Defines the file path of the obesity dataset used to train the models.
DATA_FILE = DATA_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"

# Defines where the best trained machine learning pipeline will be saved.
BEST_MODEL_FILE = MODEL_DIR / "best_pipeline.joblib"

# Defines where model information and configuration metadata will be saved.
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# Defines where the performance comparison of different models will be saved.
COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"


# Sets a fixed random seed to make randomized operations more reproducible.
RANDOM_STATE = 42

# Allocates 20% of the dataset for testing and the remaining 80% for training.
TEST_SIZE = 0.20

# Sets the number of folds used for cross-validation when evaluating models.
CV_FOLDS = 5

# Sets the number of nearest neighbors used by SMOTE to generate synthetic samples.
SMOTE_K_NEIGHBORS = 5

# Specifies the number of features to retain during feature selection.
N_FEATURES_TO_SELECT = 12

# Sets the number of samples used for SHAP-based model explanations.
SHAP_SAMPLE_SIZE = 300

# Limits hyperparameter tuning to the two strongest baseline candidates.
TOP_MODELS_TO_TUNE = 2

# Sets the number of randomized parameter combinations tested per candidate.
RANDOM_SEARCH_ITERATIONS = 8

# Specifies the column containing the obesity classification labels in the dataset.
TARGET_COLUMN = "NObeyesdad"


# Maps the original dataset labels to integer IDs for machine learning.
# The integer IDs represent the seven obesity classification categories.
TARGET_RAW_TO_ID = {
    "Insufficient_Weight": 0,  # Represents insufficient weight.
    "Normal_Weight": 1,        # Represents normal weight.
    "Overweight_Level_I": 2,   # Represents Overweight Level I.
    "Overweight_Level_II": 3,  # Represents Overweight Level II.
    "Obesity_Type_I": 4,       # Represents Obesity Type I.
    "Obesity_Type_II": 5,      # Represents Obesity Type II.
    "Obesity_Type_III": 6,     # Represents Obesity Type III.
}


# Stores readable class labels for displaying prediction results in the application.
# These labels correspond to the integer IDs defined in TARGET_RAW_TO_ID.
TARGET_CLASSES = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight Level I",
    "Overweight Level II",
    "Obesity Type I",
    "Obesity Type II",
    "Obesity Type III",
]


# Lists numerical features that can be treated as continuous or numeric inputs.
CONTINUOUS_FEATURES = [
    "Age",     # The person's age.
    "Height",  # The person's height.
    "Weight",  # The person's weight.
    "FCVC",    # Frequency of vegetable consumption.
    "NCP",     # Number of main meals consumed daily.
    "CH2O",    # Daily water consumption.
    "FAF",     # Frequency of physical activity.
    "TUE",     # Time spent using technological devices.
]


# Lists features that represent two possible categories.
BINARY_FEATURES = [
    "Gender",                         # Gender category.
    "family_history_with_overweight", # Whether the family has a history of overweight.
    "FAVC",                           # Whether the person frequently consumes high-calorie food.
    "SMOKE",                          # Whether the person smokes.
    "SCC",                            # Whether the person monitors calorie consumption.
]


# Lists categorical features whose values have a meaningful order.
ORDINAL_FEATURES = [
    "CAEC",  # Frequency of consuming food between meals.
    "CALC",  # Frequency of alcohol consumption.
]

# Lists categorical features whose categories have no natural order.
NOMINAL_FEATURES = [
    "MTRANS",  # The person's main mode of transportation.
]


# Defines the allowed category values for each binary feature, in the same order as BINARY_FEATURES.
BINARY_CATEGORIES = [
    ["Female", "Male"],  # Allowed Gender categories.
    ["no", "yes"],       # Allowed family history categories.
    ["no", "yes"],       # Allowed FAVC categories.
    ["no", "yes"],       # Allowed SMOKE categories.
    ["no", "yes"],       # Allowed SCC categories.
]


# Defines the ordered category values for each ordinal feature.
# The order allows preprocessing methods to represent the categories by their ranking.
ORDINAL_CATEGORIES = [
    ["no", "Sometimes", "Frequently", "Always"],  # Categories for CAEC.
    ["no", "Sometimes", "Frequently", "Always"],  # Categories for CALC.
]


# Combines all feature lists into one list containing every model input feature.
ALL_FEATURES = (
    CONTINUOUS_FEATURES
    + BINARY_FEATURES
    + ORDINAL_FEATURES
    + NOMINAL_FEATURES
)


# Defines acceptable input ranges for numerical fields in the web application.
# These limits help prevent obviously invalid values from being submitted.
VALIDATION_RANGES = {
    "Age": (10.0, 100.0),    # Accepts ages from 10 to 100 years.
    "Height": (1.20, 2.20), # Accepts heights from 1.20 to 2.20 metres.
    "Weight": (25.0, 250.0), # Accepts weights from 25 to 250 kilograms.
    "FCVC": (1.0, 5.0),     # Accepts vegetable consumption scores from 1 to 5.
    "NCP": (1.0, 8.0),      # Accepts daily main meal counts from 1 to 8.
    "CH2O": (0.0, 8.0),     # Accepts water consumption values from 0 to 8.
    "FAF": (0.0, 5.0),      # Accepts physical activity scores from 0 to 5.
    "TUE": (0.0, 12.0),     # Accepts technology-use values from 0 to 12 hours.
}