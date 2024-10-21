# harp

This project was modified based on the HARP project by the team "GPT is all you need."

## Table of Contents
- [Requirements and Dependencies](#Requirements and Dependencies)
- [Regression Prediction](#regression-prediction)
- [Classification Prediction](#classification-prediction)
- [Merging Prediction Files](#merging-prediction-files)
- [Additional Notes](#additional-notes)

---

## Requirements and Dependencies

You can install the required packages for running this project using:

```bash
pip install -r requirements.txt
```

## Regression Prediction

1. Modify the following parameters in `src/config`:

   ```
   ...
   TASK = 'regression'
   ...
   SUBTASK = 'inference'
   ...
   harp_path = # Replace with the absolute path to the harp project on your machine
   ...
   model_path = join(harp_path, 'models/submission/reg.pth')
   ```

2. Run the regression model:

   ```
   python main.py
   ```

3. The regression output will be saved as a CSV file(`src/regression.csv`), containing predicted values for each input data point.

------

## Classification Prediction

1. Modify the following parameters in `src/config`:

   ```
   ...
   TASK = 'class'
   ...
   SUBTASK = 'inference'
   ...
   harp_path = # Replace with the absolute path to the harp project on your machine
   ...
   model_path = join(harp_path, 'models/submission/class.pth')
   ```

2. Run the classification model:

   ```
   python main.py
   ```

3. The regression output will be saved as a CSV file('src/class.csv'), containing predicted values for each input data point.

------

## Merging Prediction Files

1. Run the following command:

   ```
   python combine_class_reg.py
   ```

2. The output will be a merged result of `regression.csv` and `class.csv` files, saved to `src/output.csv`. This CSV file will be the final submission file.

------

## Additional Notes

The version we are submitting on Kaggle relies on results produced by some models trained earlier. Unfortunately, we lost the corresponding model files. Therefore, we used a previously submitted CSV file (`history/submission_9_20.csv`) to help generate the final results.