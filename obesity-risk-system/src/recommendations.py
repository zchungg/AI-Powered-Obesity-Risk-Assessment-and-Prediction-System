from __future__ import annotations

# Dict and List describe the types of dictionaries and lists used in this file.
from typing import Dict, List


# Convert technical dataset column names into labels that are easier for users to understand.
FRIENDLY_NAMES = {
    "Age": "Age",
    "Height": "Height",
    "Weight": "Weight",
    "FCVC": "Vegetable consumption",
    "NCP": "Main meals per day",
    "CH2O": "Daily water consumption",
    "FAF": "Physical activity frequency",
    "TUE": "Technology usage time",
    "Gender": "Gender",
    "family_history_with_overweight": "Family history of overweight",
    "FAVC": "Frequent high-calorie food consumption",
    "SMOKE": "Smoking",
    "SCC": "Calorie-monitoring behaviour",
    "CAEC": "Food consumption between meals",
    "CALC": "Alcohol consumption",
    "MTRANS": "Main mode of transportation",
}


# Convert a yes/no answer into a Boolean value (True or False).
def _yes_no(value: str) -> bool:
    # Remove surrounding spaces and ignore uppercase/lowercase differences.
    return str(value).strip().lower() == "yes"


# Generate lifestyle recommendations based on the user's submitted answers.
def generate_recommendations(row: Dict) -> List[Dict[str, str]]:
    """Generate non-diagnostic lifestyle suggestions from user inputs."""

    # Store recommendations here; each one contains an area and suggestion.
    recs: List[Dict[str, str]] = []

    # Suggest gradually increasing movement when reported activity is low.
    if float(row["FAF"]) < 1.0:
        recs.append({
            "area": "Physical activity",
            "text": (
                "Consider gradually adding regular movement or physical "
                "activity to your routine, based on what is comfortable for you."
            ),
        })

    # Give a different suggestion when activity is higher but still below 2.
    elif float(row["FAF"]) < 2.0:
        recs.append({
            "area": "Physical activity",
            "text": (
                "You may benefit from keeping a consistent physical-activity "
                "routine and increasing it gradually when appropriate."
            ),
        })

    # Suggest eating vegetables more consistently when the recorded value is low.
    if float(row["FCVC"]) < 2.0:
        recs.append({
            "area": "Diet",
            "text": (
                "Consider including vegetables more consistently across your meals."
            ),
        })

    # Suggest balancing food choices when the user reports frequent high-calorie food consumption.
    if _yes_no(row["FAVC"]):
        recs.append({
            "area": "Diet",
            "text": (
                "Consider reducing how frequently you choose high-calorie foods "
                "and balancing meals with nutrient-dense options."
            ),
        })

    # Suggest planning snacks when the user often eats between meals.
    if row["CAEC"] in {"Frequently", "Always"}:
        recs.append({
            "area": "Eating pattern",
            "text": (
                "Consider planning snacks intentionally and choosing filling, "
                "nutrient-dense options when eating between meals."
            ),
        })

    # Suggest reviewing fluid intake when the recorded water-consumption value is low.
    if float(row["CH2O"]) < 2.0:
        recs.append({
            "area": "Hydration",
            "text": (
                "Try to maintain regular fluid intake throughout the day, "
                "according to your individual needs."
            ),
        })

    # Suggest screen breaks and movement when reported technology usage is high.
    if float(row["TUE"]) > 2.0:
        recs.append({
            "area": "Screen time",
            "text": (
                "Consider taking regular breaks from prolonged screen use "
                "and replacing some sedentary time with movement."
            ),
        })

    # Suggest moderating alcohol consumption when the user reports frequent use.
    if row["CALC"] in {"Frequently", "Always"}:
        recs.append({
            "area": "Alcohol",
            "text": (
                "Consider moderating alcohol consumption and following "
                "applicable health guidance for your situation."
            ),
        })

    # Offer support-seeking advice if the user reports smoking.
    if _yes_no(row["SMOKE"]):
        recs.append({
            "area": "Smoking",
            "text": (
                "If you smoke, consider speaking with an appropriate health "
                "professional about support for reducing or quitting."
            ),
        })

    # Suggest discussing meal regularity if fewer than two main meals are reported.
    if float(row["NCP"]) < 2.0:
        recs.append({
            "area": "Meal pattern",
            "text": (
                "A more consistent meal pattern may be worth discussing "
                "with a qualified professional if your current schedule is irregular."
            ),
        })

    # If no conditions matched, provide a general healthy-lifestyle suggestion.
    if not recs:
        recs.append({
            "area": "General",
            "text": (
                "Continue maintaining balanced eating habits, regular movement, "
                "adequate rest, and other healthy routines that work for you."
            ),
        })

    # Return all recommendations so the application can display them to the user.
    return recs